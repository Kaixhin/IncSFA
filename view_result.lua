local hdf5 = require 'hdf5'
local gnuplot = require 'gnuplot'

if not paths.filep('robotdata.h5') then
  os.execute('unzip robotdata.h5.zip')
end
local h5file = hdf5.open('robotdata.h5', 'r')
local DATA = h5file:read('/DATA'):all():t()
local EP_LEN = h5file:read('/EP_LEN'):all():squeeze()
local EP_START_POS = h5file:read('/EP_START_POS'):all():squeeze()
local SZ = h5file:read('/SZ'):all():squeeze()
h5file:close()

local feature_saved = torch.load('feature_saved.t7')
local V = feature_saved.V
local W = feature_saved.W
local S = feature_saved.S
local x_mean = feature_saved.x_mean
local NUMSF = W:size(2)

local y_test_big = torch.Tensor(25 - 6 + 1, torch.max(EP_LEN), NUMSF)
for j = 6, 25 do
  for k = 1, EP_LEN[j] do
    local test_cnt = EP_START_POS[j] + k - 1

    local X = DATA[{{}, {test_cnt}}] - x_mean
    local x_white = S:t() * X

    for m = 1, NUMSF do
      y_test_big[j - 6 + 1][k][m] = W[{{}, {m}}]:t() * x_white
    end
  end
end

local feat_plots = {}
for j = 6, 25 do
  feat_plots[#feat_plots + 1] = {'', y_test_big[j - 6 + 1][{{}, {1}}]:squeeze(), y_test_big[j - 6 + 1][{{}, {2}}]:squeeze(), y_test_big[j - 6 + 1][{{}, {3}}]:squeeze()}
end
gnuplot.figure(1)
gnuplot.scatter3(table.unpack(feat_plots))
gnuplot.title('Embedded Data on First Three Features')
gnuplot.xlabel('F1')
gnuplot.ylabel('F2')
gnuplot.zlabel('F3')

for myep = 6, 25 do
  print(myep)
  for k = 1, EP_LEN[myep] do
    local imgin = DATA[{{}, {EP_START_POS[myep] + k - 1}}]
    gnuplot.figure(2)
    gnuplot.imagesc(imgin:view(SZ[2], SZ[1]))

    gnuplot.figure(1)
    feat_plots[#feat_plots + 1] = {'O', torch.Tensor({y_test_big[myep - 6 + 1][k][1]}), torch.Tensor({y_test_big[myep - 6 + 1][k][2]}), torch.Tensor({y_test_big[myep - 6 + 1][k][3]})}
    gnuplot.scatter3(table.unpack(feat_plots))
    gnuplot.title('Embedded Data on First Three Features')
    gnuplot.xlabel('F1')
    gnuplot.ylabel('F2')
    gnuplot.zlabel('F3')
    feat_plots[#feat_plots] = nil
  end
end
