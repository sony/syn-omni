import logging
import os
import sys

from transformers import HfArgumentParser, set_seed

from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.collator import build_collator
from tevatron.retriever.datasets import build_train_datasets
from tevatron.retriever.modeling import Qwen2_5OmniProcessor, build_model
from tevatron.retriever.trainer import build_trainer
from tevatron.utils.dist_utils import barrier, is_main, print_master
from tevatron.utils.log_utils import save_parameters, setup_rank_logging

logging.getLogger().setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    training_component = training_args.training_component
    print_master(
        f"Training method: {training_component}. Proper collator, Model, and Trainer will be chosen"
    )

    if (
        os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir)
        and training_args.do_train and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use "
            "--overwrite_output_dir to overcome."
        )

    # Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    log_path = setup_rank_logging(training_args)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # logger.info("Training/evaluation parameters %s", training_args)
    # logger.info("MODEL parameters %s", model_args)
    print_master(f"[logging] log path: {log_path}")
    set_seed(training_args.seed)

    # save parameters
    if is_main():
        save_parameters(training_args.output_dir, model_args, data_args, training_args)

    # load model and processor
    if is_main():
        processor = Qwen2_5OmniProcessor.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,
        )
        model = build_model(training_component, model_args, training_args)
    barrier()
    logger.info("main process: loaded model")
    print("main process: loaded model")

    if not is_main():
        processor = Qwen2_5OmniProcessor.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,
        )
        model = build_model(training_component, model_args, training_args)
    barrier()
    logger.info("all sub processes: loaded model")
    print("all sub processes: loaded model")

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"

    # trainable params.
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
    print_master(f"\nTotal trainable parameters: {total:,}")
    logger.info(f"\nTotal trainable parameters: {total:,}")

    # load datasets
    train_dataset = build_train_datasets(data_args)
    collator = build_collator(training_component, data_args, processor)
    trainer = build_trainer(
        training_component,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )
    logger.info("training: all ready.")
    print("training: all ready.")
    checkpoint_dir = os.path.join(training_args.output_dir, "checkpoints")
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()
    if trainer.is_world_process_zero():
        processor.save_pretrained(checkpoint_dir)


if __name__ == "__main__":
    main()
