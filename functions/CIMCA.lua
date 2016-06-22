local function CIMCA(W, lr, gamma, input)
  local num_comps = W:size(2)
  
  W[{{}, {1}}] = (1 - lr) * W[{{}, {1}}] - lr * input * W[{{}, {1}}]:t() * input
  W[{{}, {1}}]:div(torch.norm(W[{{}, {1}}]))

  local L = torch.Tensor(W:size(1), 1)
  for i = 2, num_comps do
    L:zero()
    for j = 1, i do
      L = L + W[{{}, {j}}] * W[{{}, {i}}]:t() * W[{{}, {j}}]
    end
    
    W[{{}, {i}}] = (1 - lr) * W[{{}, {i}}] - lr * (input * W[{{}, {i}}]:t() * input + gamma * L)
    W[{{}, {i}}]:div(torch.norm(W[{{}, {i}}]))
  end

  return W
end

return CIMCA
