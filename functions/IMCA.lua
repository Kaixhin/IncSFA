local function IMCA(W, lr, gamma, input)
  local C = input * input:t()

  W[{{}, {1}}] = 1.5 * W[{{}, {1}}] - lr * C * W[{{}, {1}}] - lr * torch.dot(W[{{}, {1}}]:t(), W[{{}, {1}}]) * W[{{}, {1}}]
  C = C + gamma * (torch.dot(W[{{}, {1}}], W[{{}, {1}}]:t()) / torch.dot(W[{{}, {1}}]:t(), W[{{}, {1}}]))

  local L = torch.Tensor(W:size(1), 1)
  for i = 2, W:size(2) do
    L:zero()
    for j = 1, i do
      L = L + torch.dot(W[{{}, {j}}], W[{{}, {i}}]) * W[{{}, {j}}]
    end
    
    W[{{}, {i}}] = 1.5 * W[{{}, {i}}] - lr * C * W[{{}, {i}}] - lr * torch.dot(W[{{}, {i}}]:t(), W[{{}, {i}}]) * W[{{}, {i}}]

    C = C + gamma * (torch.dot(W[{{}, {i}}], W[{{}, {i}}]:t()) / torch.dot(W[{{}, {i}}]:t(), W[{{}, {i}}]))
  end

  return W
end

return IMCA
