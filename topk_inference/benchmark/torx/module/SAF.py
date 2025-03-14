# Copyright 2019 The PytorX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn


class SAF(nn.Module):

    def __init__(self, G_shape, p_SA00=0.1, p_SA01=0.1, p_SA10=0.1, p_SA11=0.1, G_SA00=3e-3, G_SA01=1e-3, G_SA10=2e-3, G_SA11=3e-6):
        super(SAF, self).__init__()
        '''
        This module performs the Stuck-At-Fault (SAF) non-ideal effect injection.
            Args:
                G_shape (tensor.size): crossbar array size.
                p_SA00 (FP): Stuck-at-Fault rate at 00 (range from 0 to 1).
                p_SA11 (FP): Stuck-at-Fault rate at 11 (range from 0 to 1).
                p_SA01 (FP): Stuck-at-Fault rate at 01 (range from 0 to 1).
                p_SA10 (FP): Stuck-at-Fault rate at 10 (range from 0 to 1).
                G_SA00 (FP): Stuck-at-Fault conductance at 00 (in unit of S).
                G_SA01 (FP): Stuck-at-Fault conductance at 01 (in unit of S).
                G_SA10 (FP): Stuck-at-Fault conductance at 10 (in unit of S).
                G_SA11 (FP): Stuck-at-Fault conductance at 11 (in unit of S).
        '''
        # stuck at 00 leads to high conductance
        self.p_SA00 = nn.Parameter(torch.Tensor(
            [p_SA00]), requires_grad=False)  # probability of SA00
        self.G_SA00 = G_SA00
        # stuck at 01 leads to high conductance
        self.p_SA01 = nn.Parameter(torch.Tensor(
            [p_SA01]), requires_grad=False)  # probability of SA01
        self.G_SA01 = G_SA01
        # stuck at 10 leads to high conductance
        self.p_SA10 = nn.Parameter(torch.Tensor(
            [p_SA10]), requires_grad=False)  # probability of SA10
        self.G_SA10 = G_SA10
        # stuck at 11 leads to high conductance
        self.p_SA11 = nn.Parameter(torch.Tensor(
            [p_SA11]), requires_grad=False)  # probability of SA11
        self.G_SA11 = G_SA11

        assert (
            self.p_SA00+self.p_SA11+self.p_SA01+self.p_SA10) <= 1, 'The sum of probability of SA00, SA01, SA10, SA11 is greater than 1 !!'

        # initialize a random mask
        # TODO: maybe change the SAF profile to uint8 format to avoid calculating the SAF defect
        # state on-the-fly, for simulation speedup. However the current setup has higher configurability
        # to simulate the real-time SAF state if there is run-time change .
        self.p_state = nn.Parameter(torch.Tensor(G_shape), requires_grad=False)
        self.update_SAF_profile()  # init the SAF distribution profile

    def forward(self, input):
        '''
        The forward function alter the elements that indexed by p_state to the defected conductance,
        and mask the gradient of those defect cells owing to the auto-differentiation.
        '''
        output = Inject_SAF(input, self.p_state, self.p_SA00, self.p_SA01, self.p_SA10,
                            self.p_SA11, self.G_SA00, self.G_SA01, self.G_SA10, self.G_SA11)
        return output

    def index_SA00(self):
        return self.p_state.le(self.p_SA00)

    def index_SA01(self):
        return self.p_state.le(self.p_SA00 + self.p_SA01) & self.p_state.gt(self.p_SA00)

    def index_SA10(self):
        return self.p_state.le(self.p_SA00 + self.p_SA01 + self.p_SA10) & self.p_state.gt( self.p_SA00 + self.p_SA01)

    def index_SA11(self):
        return self.p_state.gt(1-self.p_SA11)

    def update_SAF_profile(self, dist='uniform'):
        if dist == 'uniform':
            self.p_state.data.uniform_()  # update the SAF distribution.
        return


class _SAF(torch.autograd.Function):
    '''
    This autograd function performs the gradient mask for the weight
    element with Stuck-at-Fault defects, where those weights will not
    be updated during backprop through gradient masking.

    Args:
        input (Tensor): weight tensor in FP32
        p_state (Tensor): probability tensor for indicating the SAF state
        w.r.t the preset SA0/1 rate (i.e., p_SA00 and p_SA11).
        p_SA00 (FP): Stuck-at-Fault rate at 0 (range from 0 to 1).
        p_SA11 (FP): Stuck-at-Fault rate at 1 (range from 0 to 1).
        G_SA0 (FP): Stuck-at-Fault conductance at 0 (in unit of S).
        G_SA1 (FP): Stuck-at-Fault conductance at 1 (in unit of S).
    '''

    @staticmethod
    def forward(ctx, input, p_state, p_SA00, p_SA01, p_SA10, p_SA11, G_SA00, G_SA01, G_SA10, G_SA11):
        # p_state is the mask
        ctx.save_for_backward(p_state, p_SA00, p_SA01, p_SA10, p_SA11)
        output = input.clone()
        output[p_state.le(p_SA00)] = G_SA00
        output[p_state.le(p_SA00 + p_SA01) & p_state.gt(p_SA00)] = G_SA01
        output[p_state.le(p_SA00 + p_SA01 + p_SA10) & p_state.gt( p_SA00 + p_SA01)] = G_SA10
        output[p_state.gt(1-p_SA11)] = G_SA11
        return output

    @staticmethod
    def backward(ctx, grad_output):
        p_state, p_SA00, p_SA01, p_SA10, p_SA11 = ctx.saved_tensors
        grad_input = grad_output.clone()
        # mask the gradient of defect cells
        grad_input[p_state.le(p_SA00) ] = 0
        grad_input[p_state.le(p_SA00 + p_SA01) & p_state.gt(p_SA00)] = 0
        grad_input[p_state.le(p_SA00 + p_SA01 + p_SA10) & p_state.gt( p_SA00 + p_SA01)]= 0
        grad_input[p_state.gt(1-p_SA11)]=0
        #print("grad input from SAF:", grad_input, grad_input.size())
        return grad_input, None, None, None, None, None, None, None, None, None


Inject_SAF = _SAF.apply


############################################################
# Testbenchs
############################################################

# pytest
def test_SAF_update_profile():
    G_shape = torch.Size([1, 3, 3, 3])
    saf_module = SAF(G_shape)
    pre_index_SA00 = saf_module.index_SA00()
    pre_index_SA01 = saf_module.index_SA01()
    pre_index_SA10 = saf_module.index_SA10()
    pre_index_SA11 = saf_module.index_SA11()
    print("SAF AT 00", pre_index_SA00)
    print("SAF AT 01", pre_index_SA01)
    print("SAF AT 10", pre_index_SA10)
    print("SAF AT 11", pre_index_SA11)
    #saf_module.update_SAF_profile()
    #post_index_SA0 = saf_module.index_SA00()
    # print((pre_index_SA0-post_index_SA0).sum())
    #assert (pre_index_SA0 -
    #        post_index_SA0).sum().item() != 0, 'SAF profile is not updated!'
    # print(saf_module.index_SA0())
    return


def test_SA0_SA1_overlap():
    '''
    ensure there is no SAF state overlap between SA0 and SA1
    '''
    G_shape = torch.Size([3, 1, 3, 3])
    saf_module = SAF(G_shape)
    index_SA0 = saf_module.index_SA0()
    index_SA1 = saf_module.index_SA1()
    assert (index_SA0 * index_SA1).sum().item() == 0, 'exist element is 1 for both SA0/1 index!'
    return


#if __name__ == '__main__':
#     test_SAF_update_profile()
#     test_SA0_SA1_overlap()
