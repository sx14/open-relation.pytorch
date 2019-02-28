require 'Dataset'

torch.manualSeed(1234)
local method = 'contrastive'

local hdf5 = require 'hdf5'

dataset_name = 'vg'


f = hdf5.open(dataset_name .. '_dataset/wordnet_with_' .. dataset_name .. '.h5', 'r')


local originalHypernyms = f:read('hypernyms'):all():add(1) -- convert to 1-based indexing
local numEntities = torch.max(originalHypernyms) 
f:close()
print("Loaded data")




local graph = require 'Graph'

-----
-- split hypernyms into train, dev, test
-----
 -- for _, hypernymType in ipairs{'trans', 'notrans'} do
 for _, hypernymType in ipairs{'trans'} do
        local methodName = method
        local hypernyms = originalHypernyms
        if hypernymType == 'trans' then
            hypernyms = graph.transitiveClosure(hypernyms)
            methodName = methodName .. '_trans'
        end

        local N_hypernyms = hypernyms:size(1)
        
        if dataset_name == 'vg' then
          splitSize = 2000
        elseif dataset_name == 'vrd' then
          splitSize = 50
        end

        

        -- shuffle randomly
        torch.manualSeed(1)
        local order = torch.randperm(N_hypernyms):long()
        local hypernyms = hypernyms:index(1, order)
        print("Building sets ...")

        local sets = {
                test = hypernyms:narrow(1, 1, splitSize),
                val = hypernyms:narrow(1, splitSize + 1, splitSize),
                train = hypernyms
                -- train = hypernyms:narrow(1, splitSize*2+ 1, N_hypernyms - 2*splitSize)
            }
        print("Done. Building Datasets ...")
        local datasets = {}
        for name, hnyms in pairs(sets) do
            datasets[name] = Dataset(numEntities, hnyms, method)
        end

        datasets.numEntities = numEntities

        -- save visualization info
        local paths = require 'paths'
        local json = require 'cjson'
        local function write_json(file, t)
            local filename = file .. '.json'
            paths.mkdir(paths.dirname(filename))
            local f = io.open(filename, 'w')
            f:write(json.encode(t))
            f:close()
        end
    
        torch.save(dataset_name .. '_dataset/' .. methodName .. '.t7', datasets)

        write_json('vis/static/' .. methodName .. '/hypernyms', datasets.train.hypernyms:totable())
end




