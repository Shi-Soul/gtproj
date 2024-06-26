import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn.functional import relu, max_pool2d
from torch.nn.init import orthogonal_,constant_
from torch.nn.functional import softmax

class Model(nn.Module):

    def __init__(self, obs_space, act_space, task_count, last_layer=256, device="cuda",inv=False):

        super().__init__()
        self.last_layer=last_layer
        self.task_count = task_count
        self.obs_space = obs_space
        self.act_space = act_space
        output_size=obs_space[0]
        for _ in range(3):
            if output_size % 2 == 1:
                output_size = (output_size + 1) / 2
            else:
                output_size /= 2
        output_size = int(output_size ** 2 * 32)
        self.block1 = ImpalaBlock(3,16)
        self.block2 = ImpalaBlock(16,32)
        self.block3 = ImpalaBlock(32,32)
        self.harmony_linear = nn.Linear(last_layer+4,last_layer,bias=False)
        self.timestep_linear = nn.Linear(1,4,bias=False)
        self.value = nn.Linear(self.last_layer,task_count)
        self.est = nn.Linear(last_layer, task_count, bias=False)
        self.actor = nn.Linear(self.last_layer,act_space*task_count,bias=False)
        self.output_size=output_size
        if inv:
            self.inv_linear = nn.Linear(2,16)
            self.fc = nn.Linear(output_size,self.last_layer-16)
        else:
            self.fc = nn.Linear(output_size,self.last_layer)
        self.lstm = nn.LSTMCell(input_size=self.last_layer,hidden_size=self.last_layer)
        self.device=device
        self.aug_data = None

    def set_augmentation_func(self, function):
        self.function = function

    def estimate_task(self, x):
        return softmax(self.est(x),dim=1)
    
    def generate_logits(self, x, estimation):
        estimation = estimation.detach().view(-1,self.task_count,1)
        logits = self.actor(x).view(-1,self.act_space,self.task_count)
        final_logits = logits.matmul(estimation).view(-1,self.act_space)
        return final_logits

    def harmony_layer(self, x, timestep):

        # timestep = self.timestep_linear(timestep)
        # x = torch.concatenate((x,timestep),dim=1)
        # return self.harmony_linear(x)
        return x

    def forward(self, x, timestep, history=None, inv=False, **kwargs):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        y = relu(x)
        x = nn.Flatten()(y)
        x = relu(self.fc(x))
        x = relu(self.harmony_layer(x,timestep))
        if type(inv) != bool:
            inv = self.inv_linear(inv)
            x = torch.concatenate((x,inv),dim=1)
        if history is not None:
            h_0, c_0 = self.lstm(x,history)
        else:
            h_0, c_0 = self.lstm(x)

        return h_0 , (h_0, c_0)
    
    def get_action(self, x, history, inv=False, reduce=True, **kwargs):
        raise NotImplementedError
        x, (h_0, c_0) = self.forward(x, history, inv=inv)
        new_logits = self.actor(x)
        cate_o = Categorical(logits=new_logits)
        if reduce:
            new_logits = torch.where(cate_o.probs < .05, -1e+8, new_logits)
            cate_o = Categorical(logits=new_logits)
        act, log_prob = self.sample(new_logits)
        return act, (h_0, c_0), cate_o.entropy()

    def sample_act_and_value(self, x, timestep, history, **kwargs): 

        x, (h_0, c_0) = self.forward(x, timestep, history, kwargs["inv"])
        estimation = self.estimate_task(x)
        logits = self.generate_logits(x,estimation)
        act, log_prob = self.sample(logits)
        value = self.value(x)
        return act, log_prob, value, (h_0, c_0)
    
    def lstm_layer(self, x, timestep, history, **kwargs):

        x, (h_0, c_0) = self.forward(x, timestep, history, **kwargs)
        return (h_0, c_0)
    
    def check_action_and_value(self, x, timestep, act, history, inv=False, **kwargs):

        x, (h_0, c_0) = self.forward(x, timestep, history, inv=inv)
        estimation = self.estimate_task(x)
        logits = self.generate_logits(x,estimation)
        value = self.value(x)
        log_prob, entropy = self.log_prob(logits, act)
        return log_prob, entropy, value, estimation

    def sample(self, logits):

        cate_o = Categorical(logits=logits)
        act = cate_o.sample()
        log_prob = cate_o.log_prob(act)
        return act, log_prob
    
    def get_value_with_augmentation(self, x, history, timestep, **kwargs):

        shape = x.shape[1:]
        m = kwargs["m"]
        augmented_x, self.aug_data = self.function(x, aug_data = self.aug_data, **kwargs)
        augmented_x = augmented_x.to(self.device)
        h_n, c_n = history
        h_n = h_n.view(-1,1,self.last_layer).expand(-1,m+1,self.last_layer).reshape(-1,self.last_layer)
        c_n = c_n.view(-1,1,self.last_layer).expand(-1,m+1,self.last_layer).reshape(-1,self.last_layer)
        timestep = timestep.view(-1,1,1).expand(-1,m+1,1).reshape(-1,1)
        history = (h_n, c_n)
        if type(kwargs["inv"]) != bool:
            inv = kwargs["inv"].view(-1,1,2).expand(-1,m+1,2).reshape(-1,2)
            x, _ = self.forward(augmented_x, history=history, inv=inv)
        else:
            x, _ = self.forward(augmented_x, history=history)
        x = self.harmony_layer(x,timestep)
        value = self.value(x).view(-1,m+1)
        value = torch.mean(value, dim=1)
        return value.view(-1)
    
    def get_value(self, x, timestep, history, **kwargs):

        x, _ = self.forward(x, timestep, history=history, **kwargs)
        value = self.value(x) # Batch * Tasks
        return value
    
    def log_prob(self, logits, act):

        cate_o = Categorical(logits=logits)
        log_prob = cate_o.log_prob(act)
        entropy = cate_o.entropy()
        return log_prob, entropy


class ResidualBlock(nn.Module):

    def __init__(self, in_channel):

        super().__init__()
        self.first_cnn = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.second_cnn = nn.Conv2d(in_channel, in_channel, 3, padding=1)

    def forward(self, x):

        y = relu(x)
        y = self.first_cnn(y)
        y = relu(y)
        y = self.second_cnn(y)
        return y + x

class ImpalaBlock(nn.Module):

    def __init__(self, in_channel, out_channel, max_pool_kernel=3, stride=2):

        super().__init__()
        self.stride = stride
        self.max_pool_kernel = max_pool_kernel
        self.first_cnn = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.res_block1 = ResidualBlock(out_channel)
        self.res_block2 = ResidualBlock(out_channel)

    def forward(self, x):

        x = self.first_cnn(x)
        x = max_pool2d(x, self.max_pool_kernel, padding=1, stride=self.stride)
        x = self.res_block1(x)
        return self.res_block2(x)
    


        

