# methods to help
# author: satwik kottur

import torch
import sys, json, pdb, math
sys.path.append('../')

# Initializing weights
def initializeWeights(moduleList, itype):
    assert itype=='xavier', 'Only Xavier initialization supported';

    for moduleId, module in enumerate(moduleList):
        if hasattr(module, '_modules') and len(module._modules) > 0:
            # Iterate again
            initializeWeights(module, itype);
        else:
            # Initialize weights
            name = type(module).__name__;
            # If linear or embedding
            if name == 'Embedding' or name == 'Linear':
                fanIn = module.weight.data.size(0);
                fanOut = module.weight.data.size(1);

                factor = math.sqrt(2.0/(fanIn + fanOut));
                weight = torch.randn(fanIn, fanOut) * factor;
                module.weight.data.copy_(weight);

            # If LSTMCell
            if name == 'LSTMCell':
                for name, param in module._parameters.iteritems():
                    if 'bias' in name:
                        module._parameters[name].data.fill_(0.0);
                        #print('Initialized: %s' % name)

                    else:
                        fanIn = param.size(0);
                        fanOut = param.size(1);

                        factor = math.sqrt(2.0/(fanIn + fanOut));
                        weight = torch.randn(fanIn, fanOut) * factor;
                        module._parameters[name].data.copy_(weight);
                        #print('Initialized: %s' % name)

            # Check for bias and reset
            if hasattr(module, 'bias') and type(module.bias) != bool:
                module.bias.data.fill_(0.0);