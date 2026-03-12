"""Command line entrypoint of chat."""

from mlc_llm.interface.chat import ModelConfigOverride, chat
from mlc_llm.interface.help import HELP
from mlc_llm.support.argparse import ArgumentParser


def main(argv):
    """Parse command line arguments and call `mlc_llm.interface.chat`."""
    parser = ArgumentParser("MLC LLM Chat CLI")

    parser.add_argument(
        "model",
        type=str,
        help=HELP["model"] + " (required)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_deploy"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--model-lib",
        type=str,
        default=None,
        help=HELP["model_lib"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--overrides",
        type=ModelConfigOverride.from_str,
        default="",
        help=HELP["modelconfig_overrides"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--speculative-mode",
        type=str,
        choices=["disable", "small_draft", "eagle", "medusa", "dflash"],
        default="disable",
        help=HELP["speculative_mode_serve"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--additional-models",
        type=str,
        nargs="*",
        help=HELP["additional_models_serve"],
    )
    parser.add_argument(
        "--spec-draft-length",
        type=int,
        default=0,
        help=HELP["spec_draft_length_serve"] + ' (default: "%(default)s")',
    )
    parsed = parser.parse_args(argv)

    additional_models = []
    if parsed.additional_models is not None:
        for additional_model in parsed.additional_models:
            splits = additional_model.split(",", maxsplit=1)
            if len(splits) == 2:
                additional_models.append((splits[0], splits[1]))
            else:
                additional_models.append(splits[0])

    chat(
        model=parsed.model,
        device=parsed.device,
        model_lib=parsed.model_lib,
        overrides=parsed.overrides,
        speculative_mode=parsed.speculative_mode,
        additional_models=additional_models,
        spec_draft_length=parsed.spec_draft_length,
    )
