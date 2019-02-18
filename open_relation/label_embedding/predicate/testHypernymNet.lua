require 'nngraph'
require 'nn'
require 'Dataset'
require 'HypernymScore'

local function evalClassification(dataset, model, threshold)
    local input, target = dataset:all()
    local probs = model:forward(input):double()

    local inferred = probs:le(threshold)
    local accuracy = inferred:eq(target:byte()):double():mean()
    return accuracy
end


local hyperparams = {
    D_embedding = 50,
    margin = 1,
    lr = 0.01,
    norm = 2,
    eps = 0
}
threshold = 0.757
datasetPath = 'dataset/contrastive_trans.t7'
lookupTableWeights = torch.load('weights.t7')
local datasets = torch.load(datasetPath)
local hypernymNet = nn.HypernymScore(hyperparams, datasets.numEntities, lookupTableWeights)
print(hypernymNet)
-- real_accuracy = evalClassification(datasets.test, hypernymNet, threshold)
input = {torch.Tensor({2}), torch.Tensor({1})}
probs = hypernymNet:forward(input):double()
print(probs)