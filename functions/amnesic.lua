local function amnesic(time, t1, t2, c, m)
  local U

  if time < t1 then
    U = 0
  elseif time >= t1 and time < t2 then
    U = c * (time - t1) / (t2 - t1)
  else
    U = c + (time - t2) / m
  end

  local w1 = (time - 1 - U) / time
  local w2 = (1 + U) / time

  return w1, w2
end

return amnesic
