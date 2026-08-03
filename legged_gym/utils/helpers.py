import os
import copy
import torch
import numpy as np
import random
import argparse

from rsl_rl.modules import ActorCriticTSDepth

try:
    from rich_argparse import RichHelpFormatter
    HAS_RICH_ARGPARSE = True
except ImportError:
    HAS_RICH_ARGPARSE = False

def class_to_dict(obj) -> dict:
    if not hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_load_path(root, load_run=-1, checkpoint=-1):
    """Resolve a checkpoint path under an experiment log root.

    ``checkpoint`` accepts:
      * ``-1`` / ``"latest"`` — highest ``model_<iter>.pt``
      * ``"best_tracking"`` — ``best_tracking.pt``
      * ``"best"`` — ``best_tracking.pt`` if present, otherwise legacy ``best.pt``
      * integer / digit string — ``model_<iter>.pt``
    """
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except Exception:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1 or load_run is None:
        run_dir = last_run
    else:
        run_dir = load_run if os.path.isabs(str(load_run)) else os.path.join(root, load_run)

    # Prefer the shared resolver so eval CLI strings and training ints share one path.
    from legged_gym.scripts.eval.ckpt_utils import resolve_checkpoint_path
    return resolve_checkpoint_path(run_dir, checkpoint)


# MoE-CTS interleaves one student environment every four environments.  A
# one-environment debug run therefore has no teacher role at all and fails the
# reference split's exact-count checks.  Keep the exception task-specific so
# existing debug runs retain their one-environment behavior.
_MOE_CTS_DEBUG_TASKS = frozenset(("go2_moects",))
_MOE_CTS_DEBUG_NUM_ENVS = 4


def debug_num_envs_for_task(task):
    """Return the smallest debug batch that is valid for ``task``.

    Most tasks intentionally keep the historical ``--debug`` behavior of one
    environment.  MoE-CTS uses a 3:1 interleaved teacher/student split, so its
    smallest exact batch is four environments (three teachers and one
    student).
    """
    return _MOE_CTS_DEBUG_NUM_ENVS if task in _MOE_CTS_DEBUG_TASKS else 1


def update_cfg_from_args(env_cfg, cfg_train, args):
    """Override some configuration parameters from command line arguments
       Called in task_registry.py/make_env()

    Args:
        env_cfg : environment configuration
        cfg_train : training configuration
        args : command line arguments

    Returns:
        env_cfg : updated environment configuration
        cfg_train : updated training configuration
    """
    # single source of truth for the training seed: a CLI --seed override flows
    # into BOTH train_cfg.seed and env_cfg.seed (task_registry also copies
    # train_cfg.seed -> env_cfg.seed, but make_env resolves env_cfg before
    # make_alg_runner touches train_cfg, so set it on whichever cfg we were given).
    seed = getattr(args, "seed", None)
    if seed is not None:
        if env_cfg is not None:
            env_cfg.seed = seed
        if cfg_train is not None:
            cfg_train.seed = seed

    # environment parameters
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.debug:
            env_cfg.env.debug = args.debug
        if args.motion_file is not None:
            env_cfg.env.motion_file = args.motion_file
        if args.num_student is not None:
            env_cfg.env.num_camera_envs = args.num_student
            
    # training parameters
    if cfg_train is not None:
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.sync_wandb:
            cfg_train.runner.sync_wandb = args.sync_wandb
        if args.ckpt is not None:
            cfg_train.runner.checkpoint = args.ckpt
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run

    return env_cfg, cfg_train

def get_args():
    """Parse command line arguments

    Returns:
        args: parsed command line arguments
    """
    # Use RichHelpFormatter for colored help if available
    formatter_class = RichHelpFormatter if HAS_RICH_ARGPARSE else argparse.HelpFormatter
    
    parser = argparse.ArgumentParser(
        formatter_class=formatter_class,
        description="LeggedGym-Ex - Train legged robots with reinforcement learning",
        epilog="For more information, visit: https://github.com/lupinjia/LeggedGym-Ex"
    )
    parser.add_argument('--task',           type=str, default='go2', help="task name")
    parser.add_argument('--seed',           type=int, default=None, help="training seed; overrides train_cfg.seed and env_cfg.seed (single source)")
    parser.add_argument('--headless',       action='store_true', default=False, help="enable visualization by default")
    parser.add_argument('--cpu',            action='store_true', default=False, help="use CPU instead of CUDA")
    parser.add_argument('--num_envs',       type=int, default=None, help="number of parallel environments")
    parser.add_argument('--max_iterations', type=int, default=None, help="max number of training iterations")
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument('--resume',     action='store_true', default=False, help="resume training from specified checkpoint")
    init_group.add_argument('--init_checkpoint', type=str, default=None,
                            help="strict V7 flat model_1000.pt HIM policy initialization (not resume)")
    parser.add_argument('--sync_wandb',     action='store_true', default=False, help="synchronize training log with wandb")
    parser.add_argument('--export_onnx',    action='store_true', default=False, help="export policy as onnx (besides jit)")
    parser.add_argument('--debug',          action='store_true', default=False, help="enable debug mode")
    parser.add_argument('--dashboard',      action='store_true', default=True,
                        help="publish completed V5 UED curriculum stages to the local Curriculum Atlas "
                             "(default: on; harmless no-op for non-UED tasks and when no server is listening)")
    parser.add_argument('--no_dashboard',   action='store_true', default=False,
                        help="disable Curriculum Atlas publishing (overrides --dashboard / $LPACRL_DASHBOARD)")
    parser.add_argument('--dashboard_url',  type=str, default=None,
                        help="Curriculum Atlas base URL (default: $LPACRL_DASHBOARD_URL or http://127.0.0.1:8765)")
    parser.add_argument('--dashboard_run_id', type=str, default=None,
                        help="optional persistent Curriculum Atlas run id")
    parser.add_argument('--load_run',       type=str, default=None, help="run to load, default: last run")
    parser.add_argument('--continue_log_dir', type=str, default=None,
                        help="append logs/checkpoints to an existing run directory (requires --resume)")
    parser.add_argument('--ckpt',           type=str, default='-1',
                        help="checkpoint: best_tracking, best, latest, or iteration (-1 = latest)")
    parser.add_argument('--use_joystick',   action='store_true', default=False, help="use joystick to provide commands")
    parser.add_argument('--joystick_type',  type=str, default='xbox', help="type of joystick: xbox, switch")
    parser.add_argument('--follow_robot',   action='store_true', default=False, help="whether the camera follows the robot during play")
    parser.add_argument('--viewer',         type=str, default='native', choices=['native', 'viser'], 
                        help="viewer backend: 'native' for simulator viewer, 'viser' for web-based 3D viewer")
    parser.add_argument('--viser_port',     type=int, default=8080, help="port for viser web server (only used with --viewer viser)")
    parser.add_argument('--terrain',        type=str, default='flat',
                        choices=[
                            'flat', 'bumpy', 'course', 'rough', 'train',
                            'taxonomy', 'showcase', 'moe',
                            'v6', 'v6_frontier', 'frontier', 'v6_full',
                        ],
                        help="play terrain override (user-chosen, independent of the model): 'flat', mildly 'bumpy', a rough 'course' with stairs, 'rough' for the full ETH curriculum terrain on ANY model, 'train' to deliberately reproduce the checkpoint's own training terrain, 'taxonomy'/'showcase' for the LP-ACRL 6x4 exhibit, 'moe' for the go2_moects terrain bank (every distinct type x 5 spaced levels), or the go2_v6_frontier V4 bank exhibit: 'v6'/'v6_frontier'/'frontier' (levels 0/2/4/6/9 x 6 families), 'v6_full' (all 10 levels x both replicas of every family). Every v6 exhibit renders the NOMINAL severity, i.e. the mean of training's terrain_replica_variation jitter (used by play.py)")
    parser.add_argument('--terrain_level',   type=int, default=None,
                        help="curriculum difficulty (0=easiest) for --terrain rough/train; independent of the checkpoint. Default: an easy, visible level.")
    parser.add_argument(
        '--v6_tile_size',
        type=str,
        default='play',
        choices=['play', 'train'],
        help=(
            "V6/frontier showcase tile envelope only (severity formulas always "
            "come from training cfg). 'play' (default): legacy 8×8 m tiles for "
            "light multi-tile Viser exhibits. 'train': copy live "
            "Go2V6FrontierCfg terrain_length/width/platform (e.g. 80 m; heavy "
            "when stacking many tiles)."
        ),
    )
    parser.add_argument('--domain_rand', action='store_true', default=False,
                        help="keep the training domain randomization and observation "
                             "noise during play. Off by default: play is for judging a "
                             "policy, and the training ranges (e.g. friction sampled "
                             "down to 0.0) make some envs skate or hop for reasons that "
                             "have nothing to do with the checkpoint.")
    parser.add_argument('--motion_file',    type=str,
                        default=None, 
                        help="motion file to load, under resources/reference_motion")
    parser.add_argument('--motion_out_dir',  type=str, default=None, help="directory to save the motion generated by the process_reference_motion script")
    parser.add_argument('--num_student',     type=int, default=None, help="[Only used in ts_depth] number of student envs to train in parallel for distillation")
    
    return parser.parse_args()

class PolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
    
    def forward(self, obs):
        return self.actor(obs)
    
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path_pt = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path_pt)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["nn_input"]
            output_names = ["nn_output"]
            dummy_input = torch.randn(1, env_cfg.env.num_observations)
            torch.onnx.export(self, dummy_input, path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterTS(torch.nn.Module):
    """Policy exporter for teacher student policies

    Attention: This module is consistent with ActorCriticTS in rsl_rl/modules/actor_critic_ts.py
               When ActorCriticTS is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.encoder = copy.deepcopy(actor_critic.history_encoder)
    
    def forward(self, obs, history):
        latent = self.encoder(history)
        x = torch.cat([obs, latent], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        export_dir = path
        os.makedirs(export_dir, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path_pt = os.path.join(export_dir, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path_pt)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(export_dir, filename)
            input_names = ["obs_input", "obs_history_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            dummy_history = torch.randn(1, env_cfg.env.num_history_obs)
            torch.onnx.export(self, (dummy_obs, dummy_history), path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterMoECTS(PolicyExporterTS):
    """Policy exporter for MoE-CTS (go2_moects).

    Same student contract as PolicyExporterTS -- actor + history encoder, fed
    (obs, history) -- but ActorCriticMoECTS concatenates the latent FIRST
    (go2_rl_gym parity, see rsl_rl/modules/actor_critic_moe_cts.py). Exporting
    with the TS order would silently produce a policy that permutes its own
    actor input.

    Attention: keep in sync with ActorCriticMoECTS.act_student.
    """
    def forward(self, obs, history):
        latent = self.encoder(history)
        x = torch.cat([latent, obs], dim=-1)
        return self.actor(x)


class PolicyExporterEE(torch.nn.Module):
    """Policy exporter for explicit estimator policies

    Attention: This module is consistent with ActorCriticEE in rsl_rl/modules/actor_critic_ee.py
               When ActorCriticEE is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator)
    
    def forward(self, obs_history):
        estimated_state = self.estimator(obs_history)
        x = torch.cat([obs_history, estimated_state], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        pt_path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(pt_path)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            onnx_path = os.path.join(path, filename)
            input_names = ["nn_input"]
            output_names = ["nn_output"]
            dummy_input = torch.randn(1, env_cfg.env.num_estimator_features)
            torch.onnx.export(self, dummy_input, onnx_path, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterWaQ(torch.nn.Module):
    """Policy exporter for DreamWaQ policies
    
    Attention: This module is consistent with ActorCriticDreamWaQ in rsl_rl/modules/actor_critic_dreamwaq.py
               When ActorCriticDreamWaQ is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.vae = copy.deepcopy(actor_critic.vae)
    
    def forward(self, obs, obs_history):
        vae_out = self.vae.inference(obs_history)
        x = torch.cat([obs, vae_out], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["obs_input", "obs_history_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            dummy_history = torch.randn(1, env_cfg.env.num_history_obs)
            torch.onnx.export(self, (dummy_obs, dummy_history), path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterDepth(torch.nn.Module):
    """Policy exporter for depth-based policies
    
    Attention: This module is consistent with ActorCriticDepth in rsl_rl/modules/actor_critic_depth.py
               When ActorCriticDepth is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic: ActorCriticTSDepth, train_cfg):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.cnn = copy.deepcopy(actor_critic.depth_history_encoder.cnn)
        self.combination_mlp = copy.deepcopy(actor_critic.depth_history_encoder.combination_mlp)
        self.rnn = copy.deepcopy(actor_critic.depth_history_encoder.rnn)
        self.latent_output_mlp = copy.deepcopy(actor_critic.depth_history_encoder.latent_output_mlp)
        # self.register_buffer("hidden_states", 
        #                       torch.zeros(train_cfg.policy.rnn_num_layers, 1, train_cfg.policy.rnn_hidden_size))
    
    def forward(self, obs, depth_obs, hidden_states):
        cnn_encoding = self.cnn(depth_obs)
        combined_input = torch.cat((obs.unsqueeze(0), cnn_encoding.unsqueeze(0)), dim=-1)
        combined_encoding = self.combination_mlp(combined_input)
        rnn_out, hidden_states = self.rnn(combined_encoding, hidden_states)
        latent_output = self.latent_output_mlp(rnn_out)
        x = torch.cat([obs, latent_output.squeeze(0)], dim=-1)
        actions = self.actor(x)
        return actions, hidden_states
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["obs_input", "depth_obs_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            depth_cfg = env_cfg.sensor.depth_camera_config
            dummy_depth_obs = torch.randn(1, depth_cfg.resolution[0]*depth_cfg.resolution[1])
            torch.onnx.export(self, (dummy_obs, dummy_depth_obs), path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()

    def forward(self, x, hidden_state, cell_state):
        out, (h, c) = self.memory(x.unsqueeze(0), (hidden_state, cell_state))
        return self.actor(out.squeeze(0)), h, c
    
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
