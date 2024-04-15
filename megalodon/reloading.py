import os
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import torch
import torch.nn.functional as F

from megalodon.config import ModelConf, TokenizerConf, logger
from megalodon.data.tokenizer import Tokenizer
from megalodon.model.mega import (
    build_model,
    Mega,
)
from megalodon.distributed import (
    init_signal_handler,
    init_torch_distributed,
    get_data_parallel_world_size,
    get_model_parallel_world_size,
    get_chunk_parallel_world_size,
    get_data_parallel_rank,
    get_model_parallel_rank,
    initialize_model_parallel,
)
from megalodon.utils import (
    get_default_half,
    get_parallel_ranks,
    get_torch_dtype
)


@dataclass
class ReloadedConf:
    world_size: int
    dp_world_size: int
    mp_world_size: int
    cp_world_size: int
    dtype: str
    is_fsdp: bool
    cfg: Dict[str, Any]

    def new_mp_world_size(self, model_parallel_size: Optional[int]):
        return (
            model_parallel_size
            if model_parallel_size is not None
            else self.mp_world_size
        )

    def new_cp_world_size(self, chunk_parallel_size: Optional[int]):
        return (
            chunk_parallel_size
            if chunk_parallel_size is not None
            else self.cp_world_size
        )

    def __post_init__(self):
        if self.is_fsdp:
            assert self.dtype == "fp32"
        else:
            assert self.mp_world_size == self.world_size // self.cp_world_size
            assert self.dp_world_size == 1


def reload_model(
    checkpoint_dir: str,
    init_distributed: bool = True,
    model_parallel_size: Optional[int] = None,
    chunk_parallel_size: Optional[int] = None,
    dtype: str = get_default_half(),
    tokenizer_path: Optional[str] = None,
) -> Tuple[Mega, Tokenizer, ReloadedConf]:
    ckpt_dir: Path = Path(checkpoint_dir)

    reloaded, tokenizer, model_cfg = reload_config_and_tokenizer(
        ckpt_dir, tokenizer_path=tokenizer_path
    )
    new_mp = reloaded.new_mp_world_size(model_parallel_size)
    new_cp = reloaded.new_cp_world_size(chunk_parallel_size)

    if init_distributed:
        init_distributed_mode(new_mp, new_cp)

    assert new_mp == get_model_parallel_world_size(), f"{new_mp} != {get_model_parallel_world_size()}"
    assert new_cp == get_chunk_parallel_world_size(), f"{new_cp} != {get_chunk_parallel_world_size()}"

    if _is_consolidated_ckpt(ckpt_dir):
        logger.info(
            f"Reloading consolidated model -- Path={ckpt_dir} -- "
            f"MP={get_model_parallel_world_size()}"
        )
        model = build_consolidated_model(model_cfg, dtype)
        load_consolidated_weights(ckpt_dir, model, reloaded, dtype)
    else:
        logger.info(
            f"Reloading FSDP model -- Path={ckpt_dir} -- "
            f"MP={get_model_parallel_world_size()}"
        )
        model = build_model(model_cfg, dtype=dtype, fp32_reduce_scatter=(dtype != 'fp32'), reshard_after_forward=True)
        load_fsdp_weights(ckpt_dir, model, reloaded)
    return model, tokenizer, reloaded


def model_ckpt_name(rank: int) -> str:
    return f"model.ckpt.{rank:05d}.pth"


def load_fsdp_weights(ckpt_dir: Path, model: Mega, reloaded: ReloadedConf):
    assert reloaded.is_fsdp
    reload_from_shards(
        {
            i: ckpt_dir / model_ckpt_name(rank=i) for i in range(reloaded.world_size)
            if get_parallel_ranks(i, reloaded.mp_world_size, reloaded.cp_world_size)[1] == 0
        },
        model.local_state_dict(),  # type: ignore
        old_ddp=reloaded.dp_world_size,
        old_mp=reloaded.mp_world_size,
        old_cp=reloaded.cp_world_size,
    )


def reshard(
    shard_paths: Union[List[Path], List[str]],
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    assert rank < world_size
    shard_paths = sorted([str(s) for s in shard_paths])
    original_world_size = len(shard_paths)
    shard_0 = torch.load(shard_paths[0], map_location="cpu")
    to_load_ids = {"first_shard_idx": original_world_size, "last_shard_idx": -1}
    fsdp_obj_info_ls: List[dict] = []
    for metadata in shard_0["meta"]["param_metadata"]:
        fsdp_path = metadata["fsdp_path"]
        params = metadata["params"]
        fsdp_obj_info: Dict[str, int] = {}  # start and end shards, with offsets
        fsdp_obj_info_ls.append(fsdp_obj_info)
        for backing_param_name, v in params.items():
            in_state_dict_key = (
                ".".join([fsdp_path, backing_param_name])
                if fsdp_path
                else backing_param_name
            )
            _, _, numels, _ = v.values()
            total_params = sum(numels)
            original_per_shard = math.ceil(total_params / original_world_size)
            assert (
                shard_0["weights"][in_state_dict_key].numel() == original_per_shard
            ), (
                shard_0["weights"][in_state_dict_key].numel(),
                original_per_shard,
            )
            new_per_shard = math.ceil(total_params / world_size)
            fsdp_obj_info["per_shard"] = new_per_shard  # nb of params per new shard
            for r, prefix in [(rank, "first"), (rank + 1, "last")]:
                param_idx = r * new_per_shard
                shard_idx = min(
                    param_idx // original_per_shard, original_world_size - 1
                )  # old shard idx
                shard_offset = param_idx - shard_idx * original_per_shard
                assert shard_offset >= 0 and shard_offset <= original_per_shard, (
                    r,
                    prefix,
                    shard_offset,
                    original_per_shard,
                    new_per_shard,
                )
                fsdp_obj_info[f"{prefix}_shard_idx"] = shard_idx
                fsdp_obj_info[f"{prefix}_offset"] = shard_offset
            to_load_ids["first_shard_idx"] = min(
                to_load_ids["first_shard_idx"], fsdp_obj_info["first_shard_idx"]
            )
            to_load_ids["last_shard_idx"] = max(
                to_load_ids["last_shard_idx"], fsdp_obj_info["last_shard_idx"]
            )
    loaded_shards = {
        shard_idx: torch.load(path, map_location="cpu")
        for shard_idx, path in enumerate(shard_paths)
        if to_load_ids["first_shard_idx"] <= shard_idx <= to_load_ids["last_shard_idx"]
    }
    local_state_dict = {}
    for fsdp_obj_idx, metadata in enumerate(shard_0["meta"]["param_metadata"]):
        fsdp_path = metadata["fsdp_path"]
        params = metadata["params"]
        for backing_param_name, v in params.items():
            in_state_dict_key = (
                ".".join([fsdp_path, backing_param_name])
                if fsdp_path
                else backing_param_name
            )
            fdsp_obj_info = fsdp_obj_info_ls[fsdp_obj_idx]
            new_per_shard = fdsp_obj_info["per_shard"]
            first_shard_idx, last_shard_idx = (
                fdsp_obj_info["first_shard_idx"],
                fdsp_obj_info["last_shard_idx"],
            )
            weights = []
            for shard_idx in range(first_shard_idx, last_shard_idx + 1):
                assert shard_idx in loaded_shards, (
                    shard_idx,
                    first_shard_idx,
                    last_shard_idx,
                    loaded_shards.keys(),
                )
                offset = (
                    0 if shard_idx > first_shard_idx else fdsp_obj_info["first_offset"]
                )
                cut = (
                    loaded_shards[shard_idx]["weights"][in_state_dict_key].numel()
                    if shard_idx < last_shard_idx
                    else fdsp_obj_info["last_offset"]
                )
                # no unpadding, since it's either cut by :cut, or we will complete it
                weights.append(
                    loaded_shards[shard_idx]["weights"][in_state_dict_key][offset:cut]
                )
            cat = torch.cat(weights, 0)
            if cat.numel() < new_per_shard:
                assert rank == world_size - 1
                cat = F.pad(cat, [0, new_per_shard - cat.numel()])
            local_state_dict[in_state_dict_key] = cat

    for param_name in shard_0["meta"]["buffer_names"]:
        local_state_dict[param_name] = shard_0["weights"][param_name]
    return local_state_dict


def reload_from_shards(
    shard_paths: Union[List[Path], List[str], Dict[int, str]],
    local_state_dict: Dict[str, Any],
    old_ddp: int,
    old_mp: int,
    old_cp: int,
    ddp_rank: Optional[int] = None,
    ddp_world_size: Optional[int] = None,
):
    ddp_rank = get_data_parallel_rank() if ddp_rank is None else ddp_rank
    ddp_world_size = get_data_parallel_world_size() if ddp_world_size is None else ddp_world_size
    mp_rank = get_model_parallel_rank()
    mp_world_size = get_model_parallel_world_size()

    logger.info(
        f"Starting reloading from shards ({old_ddp},{old_cp},{old_mp}) -> ({ddp_world_size},{mp_world_size})"
    )

    if not Path(shard_paths[0]).exists():
        raise FileNotFoundError(
            f"{shard_paths[0]} was not found, "
            f"maybe you are using a consolidated checkpoint?"
        )

    old_global_world_size = old_ddp * old_mp * old_cp
    if type(shard_paths) is not dict:
        shard_paths = sorted([str(s) for s in shard_paths])
    groups = torch.LongTensor(range(old_global_world_size)).reshape(old_ddp, old_cp, old_mp)
    assert mp_world_size == old_mp, "Do not support new MP != old MP"
    new_state_dict = reshard(
        [shard_paths[i.item()] for i in groups[:, 0, mp_rank]],
        ddp_rank,
        ddp_world_size,
    )
    for x, y in new_state_dict.items():
        if x in local_state_dict:
            local_state_dict[x].copy_(y)
        elif mp_rank == ddp_rank == 0:
            logger.warning(f"Not copying {x} not in target dict")


def get_consolidated_ckpt_path(ckpt_dir: Path, mp_rank: int = 0, mp_size: int = 1):
    if mp_size == 1:
        assert mp_rank == 0
        return ckpt_dir / "consolidated.pth"
    else:
        return ckpt_dir / f"consolidated.{mp_rank:02d}.pth"


def load_consolidated_weights(ckpt_dir: Path, model: Mega, reloaded: ReloadedConf, dtype: str):
    assert not reloaded.is_fsdp
    if dtype != reloaded.dtype:
        logger.warning(
            f"Asking dtype: {dtype} when consolidated ckpt has dtype: {reloaded.dtype}"
        )

    if get_model_parallel_world_size() != reloaded.mp_world_size:
        raise ValueError(
            f"Asking model_parallel_size: {get_model_parallel_world_size()} "
            f"when checkpoint was consolidated with model_parallel_size: "
            f"{reloaded.mp_world_size}"
        )

    consolidated_ckpt_path = get_consolidated_ckpt_path(
        ckpt_dir=ckpt_dir,
        mp_rank=get_model_parallel_rank(),
        mp_size=get_model_parallel_world_size(),
    )
    logger.info("Loading consolidated ckpt...")
    state_dict = torch.load(consolidated_ckpt_path, map_location="cpu")
    logger.info("Done loading consolidated ckpt")
    load_state_dict(model, state_dict, strict=False)
    logger.info(f"Done with state-dict reloading.")


def load_state_dict(model: Mega, state_dict: Dict, strict: bool):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if len(missing_keys) > 0:
        logger.warning(f"Missing keys when reloading: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.warning(f"Unexpected keys when reloading: {unexpected_keys}")


def build_consolidated_model(model_cfg: ModelConf, dtype: str) -> Mega:
    logger.info(
        f"Start: building consolidated model..."
    )
    model = Mega(model_cfg)
    model.to(get_torch_dtype(dtype))
    model = model.cuda()
    logger.info(
        f"Done: building consolidated model."
    )
    return model


def init_distributed_mode(model_parallel_size: int, chunk_parallel_size: int):
    # initialize signal handler
    init_signal_handler()
    # initialize distributed mode / model parallel
    logger.info("Starting init of torch.distributed...")
    is_slurm, global_rank, world_size = init_torch_distributed()
    logger.info("Done init of torch.distributed.")

    logger.info("Starting init of model parallel...")
    initialize_model_parallel(model_parallel_size, chunk_parallel_size)
    logger.info("Done init of model parallel.")

    # print env info
    if is_slurm:
        logger.info(f"ENV: {os.environ}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"NCCL version: {torch.cuda.nccl.version()}")  # type: ignore


def reload_config_and_tokenizer(
    ckpt_dir: Path,
    tokenizer_path: Optional[str] = None,
) -> Tuple[ReloadedConf, Tokenizer, ModelConf]:
    reloaded = reload_config(ckpt_dir)
    cfg = reloaded.cfg

    tokenizer_cfg: TokenizerConf = TokenizerConf.from_dict(cfg["tokenizer"])
    old_tokenizer_path = tokenizer_cfg.path
    new_tokenizer_path: str = (
        tokenizer_path if tokenizer_path is not None else old_tokenizer_path
    )
    assert Path(new_tokenizer_path).exists(), new_tokenizer_path
    tokenizer = Tokenizer(tokenizer_cfg=tokenizer_cfg)
    model_cfg: ModelConf = ModelConf.from_dict(cfg["model"])
    model_cfg.custom_bwd = False
    model_cfg.loss_parallel = False
    model_cfg.init_mode = 'none'
    # old ckpt don't have vocab_size set
    if model_cfg.vocab_size == -1:
        model_cfg.vocab_size = tokenizer.n_words
    assert model_cfg.vocab_size == tokenizer.n_words, (
        tokenizer.n_words,
        model_cfg.vocab_size,
    )
    return reloaded, tokenizer, model_cfg


def reload_config(ckpt_dir: Path) -> ReloadedConf:
    cfg_path = ckpt_dir / "config.json"
    with cfg_path.open("r") as fp:
        cfg = json.load(fp)
    if _is_consolidated_ckpt(ckpt_dir):
        consolidate_cfg_path = ckpt_dir / "consolidate_config.json"
        if not consolidate_cfg_path.exists():
            raise RuntimeError(
                f"{consolidate_cfg_path} doesn't exists, "
                f"was the checkpoint consolidated with scripts.consolidate ?"
            )
        with consolidate_cfg_path.open("r") as fp:
            consolidate_cfg = json.load(fp)
        old_mp = consolidate_cfg["model_parallel_size"]
        old_dtype = consolidate_cfg["dtype"]
        old_ddp = 1
        old_cp = 1
        old_world_size = old_mp
        is_fsdp = False
    else:
        old_mp = cfg["model_parallel_size"]
        old_cp = cfg["chunk_parallel_size"]
        old_world_size = cfg["slurm"]["world_size"]
        assert 0 < old_mp <= old_world_size
        assert old_world_size % old_mp == 0
        old_ddp = old_world_size // old_mp // old_cp
        old_dtype = "fp32"  # FSDP training in mixed precision
        is_fsdp = True

    return ReloadedConf(
        world_size=old_world_size,
        dp_world_size=old_ddp,
        mp_world_size=old_mp,
        cp_world_size=old_cp,
        cfg=cfg,
        dtype=old_dtype,
        is_fsdp=is_fsdp,
    )


def _is_consolidated_ckpt(ckpt_dir: Path):
    return (ckpt_dir / "consolidate_config.json").exists()
