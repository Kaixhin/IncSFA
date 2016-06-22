local gnuplot = require 'gnuplot'
local quadexpand = require 'functions/quadexpand'
local amnesic = require 'functions/amnesic'
local CCIPCA = require 'functions/CCIPCA'
local CIMCA = require 'functions/CIMCA'

local N = 500
local NEPS = 120
local MCA_LR = 0.01

local t = torch.linspace(1, N, N)
t:div(N):mul(2*math.pi)

local x = torch.Tensor(2, N)
x[1] = torch.sin(t) + torch.cos(11*t):pow(2)
x[2] = torch.cos(11*t)

local ts1 = torch.sin(t)
local ts2 = torch.cos(11*t)

local DIM_REDUCED = 5
local NUMSF = 2
local V = torch.rand(5, DIM_REDUCED)
V:div(torch.sum(V))
local W = torch.rand(DIM_REDUCED, NUMSF)
W:div(torch.sum(W))
local input_mean = quadexpand(x[{{}, {1}}])
local x_derv_mean = torch.Tensor(DIM_REDUCED, 1):zero()
local Vsup = torch.Tensor(DIM_REDUCED, 2):fill(1)
local slowness = torch.Tensor(NUMSF, NEPS):zero()
local fcorr = torch.Tensor(NEPS):zero()

local __, lr, input, input_expand, input_zeroed, x_prev, x_white, cov_ra, cov_upd, x_derv, gamma
local S, D, Vn
local y1 = torch.Tensor(N)
local y2 = torch.Tensor(N)

local cnt = 1
for eps = 1, NEPS do
  print(eps)
  for i = 1, N do
    __, lr = amnesic(cnt, 20, 200, 4, 5000)

    input = x[{{}, {i}}]
    
    input_expand = torch.cat({input, torch.pow(input[1], 2), torch.pow(input[2], 2), torch.Tensor({input[1] * input[2]})}, 1)

    input_mean = (1 - lr) * input_mean + lr * input_expand
    input_zeroed = input_expand - input_mean

    V, S, D, Vn = CCIPCA(V, cnt + 1, input_zeroed)

    if i > 1 then
      x_prev = x_white
    end
    x_white = S:t() * input_zeroed

    if cnt == 1 then
      cov_ra = x_white * x_white:t()
    else
      cov_upd = x_white * x_white:t()
      cov_ra = (1 - 0.005) * cov_ra + 0.005 * cov_upd
    end

    if i > 1 then
      x_derv = x_white - x_prev
      x_derv_mean = (1 - lr) * x_derv_mean + lr * x_derv
      
      Vsup = CCIPCA(Vsup, cnt + 1, x_derv - x_derv_mean)
      gamma = torch.norm(Vsup[{{}, {1}}])

      if eps > 1 then
        W = CIMCA(W, MCA_LR, gamma, x_derv - x_derv_mean)
      end
    end

    cnt = cnt + 1
  end

  for i = 1, N do
    input = x[{{}, {i}}]

    input_expand = torch.cat({input, torch.pow(input[1], 2), torch.pow(input[2], 2), torch.Tensor({input[1] * input[2]})}, 1)

    x_white = S:t() * (input_expand - input_mean)

    y1[i] = W[{{}, {1}}]:t() * x_white
    y2[i] = W[{{}, {2}}]:t() * x_white

    slowness[1][eps] = torch.mean((y1[{{2, N}}] - y1[{{1, N - 1}}]):pow(2))
    slowness[2][eps] = torch.mean((y2[{{2, N}}] - y2[{{1, N - 1}}]):pow(2))

    local Wabs1 = torch.abs(W[{{}, {1}}])
    local Wabs2 = torch.abs(W[{{}, {2}}])
    fcorr[eps] = torch.dot(Wabs1 / torch.norm(Wabs1), Wabs2 / torch.norm(Wabs2))
  end

  y1:csub(torch.mean(y1))
  y1:div(torch.max(torch.abs(y1)))

  y2:csub(torch.mean(y2))
  y2:div(torch.max(torch.abs(y2)))

  gnuplot.figure(1)
  gnuplot.plot({'Feature 1 Output', y1, '-'}, {'Ground Truth', ts1, '-'})
  gnuplot.title('Feature One Output --- Green: G.Truth')
  gnuplot.xlabel('Sequence Time')
  gnuplot.ylabel('Feature Response')
  gnuplot.figure(2)
  gnuplot.plot({'Feature 2 Output', y2, '-'}, {'Ground Truth', ts2, '-'})
  gnuplot.title('Feature Two Output --- Green: G.Truth')
  gnuplot.xlabel('Sequence Time')
  gnuplot.ylabel('Feature Response')
  gnuplot.figure(3)
  gnuplot.plot({'', slowness[1][{{1, eps}}], '-'}, {'', slowness[2][{{1, eps}}], '-'})
  gnuplot.title('Slowness of Feature One (B) and Two (G)')
  gnuplot.xlabel('Episode of Training')
  gnuplot.ylabel('Slowness')
  gnuplot.figure(4)
  gnuplot.plot({'', fcorr[{{1, eps}}], '-'})
  gnuplot.title('Correlation of the Features')
  gnuplot.xlabel('Episode of Training')
  gnuplot.ylabel('Corr.')
end
