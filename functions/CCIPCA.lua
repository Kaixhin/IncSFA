local amnesic = require './amnesic'

local function CCIPCA(V, time, input, pp, num_comp)
  if num_comp == nil then
    num_comp = V:size(2)
  end

  if pp == nil then
    pp = {20, 200, 3, 2000}
  end

  local __, lr = amnesic(time, pp[1], pp[2], pp[3], pp[4])

  local Xt = input

  local Vnorm = torch.Tensor(num_comp)
  for j = 1, num_comp do
    V[{{}, {j}}] = (1 - lr) * V[{{}, {j}}] + lr * torch.dot(V[{{}, {j}}]:t(), Xt) / torch.norm(V[{{}, {j}}]) * Xt
    Xt = Xt - torch.dot(Xt:t(), V[{{}, {j}}]) / torch.norm(V[{{}, {j}}]) * V[{{}, {j}}] / torch.norm(V[{{}, {j}}])
    Vnorm[j] = torch.norm(V[{{}, {j}}])
  end

  local order
  __, order = torch.sort(Vnorm, 1, true)
  V = V[{{}, order}]

  local Vn = torch.Tensor(V:size())
  local D = torch.Tensor(num_comp, num_comp):zero()
  for j = 1, num_comp do
    Vn[{{}, {j}}] = V[{{}, {j}}] / torch.norm(V[{{}, {j}}])
    D[{{j}, {j}}] = 1 / torch.sqrt(torch.norm(V[{{}, {j}}]))
  end

  local S = Vn * D
  
  return V, S, D, Vn
end

return CCIPCA
