import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os

class PolicyValueNet(nn.Module):
    def __init__(self, board_width, board_height, block, init_model=None, transfer_model=None, cuda=False):
        super(PolicyValueNet, self).__init__()
        print()
        print('building network ...')
        print()

        self.planes_num = 9  # 特征平面的数量 
        self.nb_block = block  #  ResNet 块的数量
        if not cuda:
            # 这些代码片段用于配置是否使用 GPU 加速
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.board_width = board_width # 棋盘宽度
        self.board_height = board_height # 棋盘高度

        self.input_states = nn.Parameter(torch.FloatTensor(1, self.planes_num, board_height, board_width))
        #  (1, self.planes_num, board_height, board_width) 的四维张量

        # self.action_fc_train, self.evaluation_fc2_train = self.network(input_states=self.input_states,
        #                                                                is_train=True)
        self.net = self.network(input_states=self.input_states,is_train=True)
        model1 = self.network(input_states=self.input_states,is_train=True)
        self.action_fc_train, self.evaluation_fc2_train = model1.forward(input_states=self.input_states)

        # self.action_fc_test, self.evaluation_fc2_test = self.network(input_states=self.input_states,
        #                                                              is_train=False)
        model2 = self.network(input_states=self.input_states,is_train=False)
        self.action_fc_test, self.evaluation_fc2_test = model2.forward(input_states=self.input_states)

        self.network_all_params = list(self.parameters())

        # 定义损失函数
        self.labels = nn.Parameter(torch.FloatTensor(1, 1))

        self.value_loss = nn.MSELoss()(self.evaluation_fc2_train, self.labels)
        # 计算策略损失
        self.mcts_probs = nn.Parameter(torch.FloatTensor(1, board_height * board_width))
        self.policy_loss = -torch.mean(torch.sum(self.mcts_probs * self.action_fc_train, dim=1))

        l2_penalty_beta = 1e-4 
        # 这里还有问题
        l2_penalty = l2_penalty_beta * torch.sum(torch.stack([torch.norm(v) for v in self.parameters()]))
        # l2_penalty = 0
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        self.learning_rate = nn.Parameter(torch.tensor([0.001], dtype=torch.float32))
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.entropy = -torch.mean(torch.sum(torch.exp(self.action_fc_test) * self.action_fc_test, dim=1))

        self.network_params = list(self.parameters())

        self.restore_params = []

        for name, param in self.named_parameters():
            if ('conv2d' in name) or ('resnet' in name) or ('bn' in name) or ('flatten_layer' in name):
                self.restore_params.append(param)

        self.init_model = init_model
        self.transfer_model = transfer_model
        print(init_model,"???????")
        if self.init_model is not None:
            self.restore_model(self.init_model)
            print('model loaded!')
        elif self.transfer_model is not None:
            self.load_state_dict(torch.load(self.transfer_model))
            print('transfer model loaded!')
        else:
            print('cannot find saved model, learn from scratch!')

        # self.action_fc_train_oppo, self.evaluation_fc2_train_oppo = self.network(input_states=self.input_states,
        #                                                                          is_train=True,
        #                                                                          label='_oppo')
        model3 = self.network(input_states=self.input_states, is_train=True,label='_oppo')   
        self.action_fc_train_oppo, self.evaluation_fc2_train_oppo = model3.forward(input_states=self.input_states)      

        # self.action_fc_test_oppo, self.evaluation_fc2_test_oppo = self.network(input_states=self.input_states,
        #                                                                        is_train=False,
        #                                                                        label='_oppo')
        model4 = self.network(input_states=self.input_states,is_train=False,label='_oppo')
        self.action_fc_test_oppo, self.evaluation_fc2_test_oppo = model4.forward(input_states=self.input_states)

        self.network_oppo_all_params = list(self.parameters())

    def policy_value_fn_random(self, board, actin_fc, evaluation_fc):
        legal_positions = board.availables
        current_state = torch.as_tensor(board.current_state().reshape(
            -1, self.planes_num, self.board_width, self.board_height).copy()).float()
       
        rotate_angle = torch.randint(1, 5, (1,))
        flip = torch.randint(0, 2, (1,))
        equi_state = torch.rot90(current_state[0], rotate_angle.item(), (1, 2))
        if flip.item():
            equi_state = torch.flip(equi_state, (2,))

        # put equi_state to network
        act_probs, value = self.policy_value(equi_state.unsqueeze(0), actin_fc, evaluation_fc)

        # get dihedral reflection or rotation back
        equi_mcts_prob = torch.flipud(torch.from_numpy(act_probs[0].reshape(self.board_height, self.board_width)).clone())
        if flip.item():
            equi_mcts_prob = torch.fliplr(equi_mcts_prob)
        equi_mcts_prob = torch.rot90(equi_mcts_prob, 4 - rotate_angle.item())
        act_probs = torch.flipud(equi_mcts_prob).flatten()

        act_probs = list(zip(legal_positions, act_probs[legal_positions].tolist()))
        return act_probs, value
    
    def policy_value(self, state_batch, actin_fc, evaluation_fc):
        '''
        input: a batch of states, actin_fc, evaluation_fc
        output: a batch of action probabilities and state values
        '''
        state_tensor = torch.tensor(state_batch)  # 转换输入状态为 PyTorch 张量
        model2 = self.network(state_tensor,is_train=False)
        log_act_probs, value = model2.forward(state_tensor)
         
        act_probs = torch.exp(log_act_probs)  # 计算指数概率
        return act_probs.detach().numpy(), value.detach().numpy()  # 转换输出为 numpy 数组


    def policy_value_fn(self, board, actin_fc, evaluation_fc): #
        '''
        input: board, actin_fc, evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        legal_positions = board.availables
        current_state = torch.as_tensor(board.current_state().reshape(
            -1, self.planes_num, self.board_width, self.board_height)).float()
        act_probs, value = self.policy_value(current_state, actin_fc, evaluation_fc)
        act_probs = list(zip(legal_positions, act_probs[0, legal_positions].tolist()))
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        winner_batch = winner_batch.reshape(-1, 1)
        
        self.optimizer.zero_grad()
        
        input_states = torch.tensor(state_batch)
        mcts_probs = torch.tensor(mcts_probs)
        winner_batch = torch.tensor(winner_batch)
        net = Network()
        action_fc, evaluation_fc2 = net.forward(input_states)
        
        action_loss = -torch.sum(mcts_probs * action_fc) / input_states.size(0)
        value_loss = torch.mean((winner_batch - evaluation_fc2) ** 2)
        entropy = -torch.sum(action_fc * torch.log(action_fc + 1e-10)) / input_states.size(0)
        
        loss = action_loss + value_loss - 0.0001 * entropy
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), entropy.item()


    def save_model(self, model_path):
        '''
        save model with ckpt form
        '''
        torch.save(self.model.state_dict(), model_path)

    def restore_model(self, model_path):
        '''
        restore model from ckpt
        '''
        
        self.net.load_state_dict(torch.load(model_path))


    def network(self, input_states, is_train, label=''):
        board_width = input_states.shape[2]
        board_height = input_states.shape[3]
        nb_block =   self.nb_block # 根据您的代码，将 nb_block 设置为 9

        return Network(board_width, board_height, nb_block, label)

class Network(nn.Module):
    def __init__(self, board_width, board_height, nb_block, label):
        super(Network, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.nb_block = nb_block
        
        self.conv1 = nn.Conv2d(9, 64, kernel_size=1, stride=1, padding=0) # 这里第一个元素从4改成了9
        self.residual_layer = self._make_residual_block(64, nb_block)

        self.action_conv = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(2)
        self.flatten_layer_1 = nn.Flatten()
        # 2*11*11
        # print(2 * board_width * board_height)
        self.action_fc = nn.Linear(2 * board_width * board_height, board_width * board_height)
        # self.action_fc = nn.Linear(2 * (board_width + 4) * (board_height + 4), board_width * board_height)
        self.evaluation_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(1)
        self.flatten_layer_2 = nn.Flatten()
        self.evaluation_fc1 = nn.Linear(board_width * board_height, 256)
        self.evaluation_fc2 = nn.Linear(256, 1)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

        self.label = label

    def forward(self, input_states):
        x = self.conv1(input_states)
        x = self.residual_layer(x)

        action_conv = self.action_conv(x)
        action_conv = self.bn_1(action_conv)
        action_conv_flat = self.flatten_layer_1(action_conv)
        # action_conv_flat 是(1, 242)，  self.action_fc 的权重矩阵的维度是 
        # action_conv_flat = action_conv.view(-1, 242)
        action_fc = self.log_softmax(self.action_fc(action_conv_flat))
        # action_conv_flat 是mat1，它展平了，action_fc是mat2
        evaluation_conv = self.evaluation_conv(x)
        evaluation_conv = self.bn_2(evaluation_conv)
        evaluation_conv_flat = self.flatten_layer_2(evaluation_conv)
        evaluation_fc1 = self.evaluation_fc1(evaluation_conv_flat)
        evaluation_fc2 = self.tanh(self.evaluation_fc2(evaluation_fc1))

        return action_fc, evaluation_fc2

    def _make_residual_block(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)   # 并按照它们在参数列表中的顺序依次执行这些层, 序列化

    

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
        
