require 'nngraph'
require 'nn'
require 'dpnn'
require 'Dataset'
require 'hdf5'
require 'config'

dataset_name = config.dataset_name
target = config.target

if dataset_name == 'vrd' then
  featureDimension = 600
else
  featureDimension = 1000
end


datasetPath = dataset_name .. '_dataset/contrastive_trans_'.. target ..'.t7'
weights = torch.load('weights_' .. dataset_name .. '_' .. target .. '.t7')


dataset = torch.load(datasetPath)
lookup = nn.LookupTable(dataset.numEntities, featureDimension)
lookup.weight = weights:double()
fs = torch.Tensor(dataset.numEntities, featureDimension)
embedding = nn.Sequential():add(lookup)
for i=1, dataset.numEntities do
  input = torch.Tensor({i})
  f = embedding:forward(input):clone()
  fs[i] = f
end

myFile = hdf5.open('label_vec_' .. dataset_name .. '_' .. target ..'.h5', 'w')

myFile:write('label_vec', torch.Tensor(fs))
myFile:close()
