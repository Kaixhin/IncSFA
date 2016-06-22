local function quadexpand(input)
  local t1 = input
  local t2 = torch.triu(input * input:t())
  t2 = t2[t2:ne(0)]

  return torch.cat(t1, t2, 1)
end

return quadexpand
