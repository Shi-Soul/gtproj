

class ImpalaConfig:
    num_envs=4
    epoch=4
    num_steps=256
    total_timesteps=int(2e7)
    minibatch=4
    gamma=.99
    gae=.95
    ent_coef=1e-2
    clip_coef=.2
    lr=2.5e-4
    load=False
    load_loc="./Tess/saved_models/territory__rooms/territory__rooms-v11.pt"
    est_coef=1
    clip_v_loss=False
    use_advantage_norm=False
    clip_v=.1
    v_coef=1
    visual=False
    check_partial_obs = False # only for debug
    a=.9
    b=1.1
    max_kl = .04
    m=0
    beta = 1e-3
    aug_coef = 1
    aug = "change_color_channel"
    burn_in=8 # Can't be higher than num_steps // minibatch
    cut_size=5
    var= 0.0001
    optimizer = "rmsprop"
    tasks = [0,0,0,0] # Length should be equal to number of envs
    focals = [4,4,16,16] # Length should be equal to number of envs
    bot_included_env = 2 # Should be lesser then env amount
    bot = 0
    task_count = 2
    vec_means = [0.4027, 1.6092, 11.1136]
    vec_dev = [0.8088, 2.4742, 7.0627]
    vec_v = [.8163, 8.7113, 173.3928]