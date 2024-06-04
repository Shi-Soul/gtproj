import argparse
import contextlib
import json
import pandas as pd
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from baselines.train.configs import SUPPORTED_SCENARIOS
from baselines.customs.policies_tess import EvalPolicy
from baselines.wrappers.downsamplepolicy_wrapper import DownsamplingPolicyWraper
from collections.abc import Iterator, Mapping
from collections import defaultdict
from meltingpot.utils.evaluation import evaluation
from meltingpot.utils.policies import policy as policy_lib


@contextlib.contextmanager
def build_focal_population(
    ckpt_paths, policy_ids, scale, substrate="pd_matrix"
) -> Iterator[Mapping[str, policy_lib.Policy]]:
    """Builds a population from the specified saved models.

    Args:
      ckpt_paths: path where agent policies are stored
      policy ids: policy ids for each agent

    Yields:
      A mapping from policy id to policy required to build focal population for evaluation.
    """
    with contextlib.ExitStack() as stack:
        yield {
            p_id: stack.enter_context(
                DownsamplingPolicyWraper(EvalPolicy(ckpt_paths, substrate), scale)
            )
            for p_id in policy_ids
        }


def run_evaluation(args):
    # scenario = "prisoners_dilemma_in_the_matrix__repeated"
    scenario = "clean_up"
    
    if args.eval_on_scenario:
        scenario = args.scenario
        
    print("Scenario: ", scenario)
    
    if "prison" in scenario:
        substrate = "pd_matrix"
        roles=['default']*2
    else:
        substrate = "clean_up"
        roles=['default']*7

    if args.create_videos:
        video_dir = args.video_dir
    else:
        video_dir = None

    policies_path = args.policies_dir
    
    policy_ids = ["shared_agent" for i in range(len(roles))]
    names_by_role = defaultdict(list)
    for i in range(len(policy_ids)):
        names_by_role[roles[i]].append(policy_ids[i])

    # Build population and evaluate
    with build_focal_population(policies_path, policy_ids, scale=1, substrate=substrate) as population:
        results = evaluation.evaluate_population(
            population=population,
            names_by_role=names_by_role,
            scenario=scenario,
            num_episodes=args.num_episodes,
            video_root=video_dir,
        )
    return results, scenario


def main():

    parser = argparse.ArgumentParser(
        description="Evaluation Script for Multi-Agent RL in Meltingpot"
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=2,
        help="Number of episodes to run evaluation",
    )
    parser.add_argument(
        "--eval_on_scenario",
        type=bool,
        default=False,
        help="Whether to evaluate on scenario. If this is False, evaluation is done on substrate",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Name of the scenario. This cannot be None when eval_on_scenario is set to True.",
    )

    # parser.add_argument(
    #     "--config_dir",
    #     type=str,
    #     help="Directory where your experiment config (params.json) is located",
    # )

    parser.add_argument(
        "--policies_dir",
        type=str,
        help="Directory where your trained polcies are located",
    )

    parser.add_argument(
        "--create_videos",
        type=bool,
        default=False,
        help="Whether to create evaluation videos",
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        help="Directory where you want to store evaluation videos",
    )
    parser.add_argument(
        "--param_sharing",
        action="store_true",
        help="Whether to share parameters across agents.",
        
    )
    parser.add_argument(
        "--expname",
        type=str,
        help="Directory where you want to store evaluation videos",
    )

    args = parser.parse_args()

    print("Evaluating with the following arguments: ", args)

    # if args.eval_on_scenario:
    #     if args.scenario is None:
    #         raise Exception(
    #             "Either set evaluate_on_scenario to False or provide a scenario name from supported scenarios"
    #         )
    #     if args.scenario not in SUPPORTED_SCENARIOS:
    #         raise Exception(
    #             "Provide a valid scenario name from supported scenarios. Supported scenarios are: ",
    #             SUPPORTED_SCENARIOS,
    #         )
    # else:
    #     print(
    #         "evaluate_on_scenario=False. Evaluating on substrate found in the config file provided."
    #     )

    results, scenario = run_evaluation(args)
    print(f"Results for {scenario}: ")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(results)
        results.to_csv(f"results_{args.expname}_{args.scenario if args.eval_on_scenario else ''}.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">>>>>>>>>BUG: ", e)
        import pdb

        pdb.post_mortem()
