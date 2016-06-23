local paths = require 'paths'
local hdf5 = require 'hdf5'
local gnuplot = require 'gnuplot'
local amnesic = require 'functions/amnesic'
local CCIPCA = require 'functions/CCIPCA'
local CIMCA = require 'functions/CIMCA'

local INCREMENTAL_VIZ = true
local NUMSF = 3
local DIM_REDUCED = 40
local MCA_LR = 0.0005
local PP = {20, 200, 2, 10000}
local MAX_NUM_EPISODE = 200

if not paths.filep('robotdata.h5') then
  os.execute('unzip robotdata.h5.zip')
end
local h5file = hdf5.open('robotdata.h5', 'r')
local DATA = h5file:read('/DATA'):all():t()
local EP_LEN = h5file:read('/EP_LEN'):all():squeeze()
local EP_START_POS = h5file:read('/EP_START_POS'):all():squeeze()
local SZ = h5file:read('/SZ'):all():squeeze()
h5file:close()
local DIM = DATA:size(1)
local NUM_EPISODES = EP_LEN:size(1)
local TEST_EPS = {1, 2, 3}
local TEST_EP_LEN = EP_LEN[{{1, 3}}]
local NUM_TEST_EPS = #TEST_EPS

local V = DATA[{{}, {1, DIM_REDUCED}}]:clone()
local W = torch.randn(DIM_REDUCED, NUMSF)
W:div(torch.sum(W))
local x_mean = DATA[{{}, {1}}] + 1
local x_derv_mean = torch.Tensor(DIM_REDUCED, 1):zero()
local Vsup = torch.Tensor(DIM_REDUCED, 1):fill(1)
local slowness = torch.Tensor(NUMSF, NUM_TEST_EPS):zero()
local fcorr = torch.Tensor(MAX_NUM_EPISODE):zero()

local cnt = 1
local slowness_history = torch.Tensor(NUM_EPISODES, NUMSF)
local curr_eps, ep_start, ep_end
local actual_index, X, __, lr, input_zeroed, S, D, Vn, x_prev, x_white, x_derv, x_derv_no_mean

print('Training begins!')
for episode = 1, NUM_EPISODES do
  curr_eps = torch.random(NUM_EPISODES)
  ep_start = EP_START_POS[curr_eps]
  ep_end = EP_START_POS[curr_eps + 1] - 1
  
  for i = 1, EP_LEN[curr_eps] do
    actual_index = EP_START_POS[curr_eps] + i - 1
    X = DATA[{{}, {actual_index}}]
    
    __, lr = amnesic(cnt, PP[1], PP[2], PP[3], PP[4])
    x_mean = (1 - lr) * x_mean + lr * X
    input_zeroed = X - x_mean
    
    V, S, D, Vn = CCIPCA(V, cnt + 1, input_zeroed, PP)

    if i > 1 then
      x_prev = x_white
    end
    x_white = S:t() * input_zeroed
    
    if i > 1 then
      x_derv = x_white - x_prev

      x_derv_mean = (1 - lr) * x_derv_mean + lr * x_derv
      x_derv_no_mean = x_derv - x_derv_mean

      Vsup = CCIPCA(Vsup, cnt + 1, x_derv_no_mean, PP)
      gamma = torch.norm(Vsup[{{}, {1}}])

      W = CIMCA(W, MCA_LR, gamma, x_derv_no_mean)
      
    end

    cnt = cnt + 1
  end

  if episode % 5 == 0 then
    print('Features saved...')
    torch.save('feature_saved.t7', {V = V, W = W, S = S, x_mean = x_mean})
  end

  if INCREMENTAL_VIZ then
    local y = torch.Tensor(NUM_TEST_EPS, torch.max(TEST_EP_LEN), NUMSF)

    for j = 1, NUM_TEST_EPS do
      for k = 1, TEST_EP_LEN[j] do
        local test_cnt = EP_START_POS[TEST_EPS[j]] + k -1
        
        X = DATA[{{}, {test_cnt}}] - x_mean
        x_white = S:t() * X

        for m = 1, NUMSF do
          y[j][k][m] = W[{{}, {m}}]:t() * x_white
        end
      end
    end

    local plot_cnt = 1
    for j = 1, NUM_TEST_EPS do
      y[j]:csub(torch.mean(y[j]))
      y[j]:div(torch.max(torch.abs(y[j])))
      
      for k = 1, NUMSF do
        gnuplot.figure(plot_cnt)
        gnuplot.plot({'', y[{{j}, {}, {k}}]:squeeze(), '-'})
        gnuplot.axis({0, TEST_EP_LEN[j], '', ''})
        gnuplot.title('Test Episode ' .. j .. ': F' .. k)
        gnuplot.xlabel('Within Episode Time')
        gnuplot.ylabel('Feat. Output')

        slowness[k][j] = torch.mean(torch.abs(y[j][{{2, TEST_EP_LEN[j]}, {k}}] - y[j][{{1, TEST_EP_LEN[j] - 1}, {k}}]))
        
        plot_cnt = plot_cnt + 1
      end
    end

    local slw_per_feat = torch.mean(slowness, 1):squeeze()

    local Wtemp = W:clone()
    local order
    slw_per_feat, order = torch.sort(slw_per_feat, 1)
    for j = 1, NUMSF do
      W[{{}, {j}}] = Wtemp[{{}, {order[j]}}]
    end

    slowness_history[episode] = slw_per_feat

    local floor_plot = math.max(episode - 10, 1)
    local slowness_plots = {}
    for j = 1, NUMSF do
      slowness_plots[#slowness_plots + 1] = {'', slowness_history[{{floor_plot, episode}, {j}}], '-'}
    end
    gnuplot.figure(NUM_TEST_EPS + NUMSF + 1)
    gnuplot.plot(table.unpack(slowness_plots))
    gnuplot.title('Slowness Measured on a Few Testing episodes')
    gnuplot.xlabel('Previous Episodes of Training')
    gnuplot.ylabel('Slowness')
    
    local temp = 0
    local temp_cnt = 0
    for j = 1, NUMSF do
      for k = 1, NUMSF do
        if j ~= k then
          temp = temp + torch.abs(torch.dot(W[{{}, {j}}] / torch.norm(W[{{}, {j}}]), W[{{}, {k}}] / torch.norm(W[{{}, {k}}])))
          temp_cnt = temp_cnt + 1
        end
      end
    end
    fcorr[episode] = temp / temp_cnt

    gnuplot.figure(NUM_TEST_EPS + NUMSF + 2)
    gnuplot.plot({'', fcorr[{{1, episode}}], '-'})
    gnuplot.title('Feature Correlation')
    gnuplot.xlabel('Episode')
    gnuplot.ylabel('Avg. Similarity')
  end
end

dofile('view_result.lua')
