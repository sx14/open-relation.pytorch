require 'nn'
require 'dpnn'

local HypernymScore, parent = torch.class('nn.HypernymScore', 'nn.Sequential')

function HypernymScore:__init(params, num_entities, weights)
    parent.__init(self)
    local lookup = nn.LookupTable(num_entities, params.D_embedding)
    if weights ~= nil then
      lookup.weight = weights:double()
    end
    
    local embedding = nn.Sequential():add(lookup)
    local embedding2 = embedding:sharedClone()
    -- self: takes two input words, outputs a probability that the first is a hypernym of the second
    self:add(nn.ParallelTable():add(embedding):add(embedding2))
    if params.symmetric then
        self:add(nn.CosineDistance())
        self:add(nn.AddConstant(1))
        self:add(nn.MulConstant(0.5))
    else
        self:add(nn.CSubTable())
        -- 1,1,1,1
        -- 2,2,2,2
        self:add(nn.AddConstant(params.eps))
        self:add(nn.ReLU()) -- max(0,(x - y))
        if params.norm > 1000 then
            self:add(nn.Max(2)) -- max(1,2,3) -> 3
        elseif params.norm == 2 then
            self:add(nn.Power(2)) -- 1,1,1,1
            self:add(nn.Sum(2))   -- 4
        elseif params.norm == 1 then
            self:add(nn.Mean(2))
        end
    end

    if USE_CUDA then
        self:cuda()
        --reshare parameters
        embedding:share(embedding2, 'weight', 'bias', 'gradWeight', 'gradBias')
    end
    self.lookupModule = lookup
end

