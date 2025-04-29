%% lane6_final_fixed_xy.m —— 修正轴向，确保y覆盖完整
clear; close all; clc
matFile = 'lc_data_20s_withpoints.mat';
idx     = 1;
L       = 200; dt = 0.1;

% --- 加载数据 ---
S = load(matFile);
if iscell(S.lc_data)
    scene = S.lc_data{idx};
else
    scene = S.lc_data(idx);
end

cars = {scene.veh_s, scene.veh_f, scene.veh_r, ...
        scene.veh_ft, scene.veh_rt, scene.veh_st};
tags = {'ego','f','r','ft','rt','st'};
cols = {[1 0 0], [.2 .2 .2], [.2 .2 .2], [0 0 1], [0.1 0.7 0.1], [1 0.6 0]};

% --- 计算坐标轴范围（排除异常值） ---
x_all = []; y_all = [];
for c = cars
    x_all = [x_all; c{1}.x(:)];
    y_all = [y_all; c{1}.y(:)];
end
x_all = x_all(~isnan(x_all) & abs(x_all)<50);
y_all = y_all(~isnan(y_all) & abs(y_all)<1000);

xLim  = [min(x_all)-5, max(x_all)+5];
yLim  = [min(y_all)-10, max(y_all)+10];

figure('Color','w'); set(gcf,'Position',[200 200 680 480])
gifName = 'lane6_fixed_xy.gif';

for k = 1:L
    clf; hold on
    for i = 1:6
        cx = cars{i}.x(k);  cy = cars{i}.y(k);
        if isnan(cx) || isnan(cy), continue; end
        plot(cy, cx, 's', 'MarkerFaceColor', cols{i}, ...
             'MarkerEdgeColor', 'k', 'MarkerSize', 8);
        text(cy, cx-0.5, tags{i}, 'HorizontalAlignment','center','FontSize',8);
    end
    title(sprintf('6-car positions | frame %d/%d (t=%.1fs)', k, L, (k-1)*dt))
    xlabel('longitudinal y (m)'); ylabel('lateral x (m)')
    xlim(yLim); ylim(xLim); set(gca,'XDir','normal'); axis equal; grid on
    drawnow

    % 写 GIF
    F = getframe(gcf);
    [A,map] = rgb2ind(frame2im(F),256);
    if k == 1
        imwrite(A,map,gifName,'gif','LoopCount',Inf,'DelayTime',dt);
    else
        imwrite(A,map,gifName,'gif','WriteMode','append','DelayTime',dt);
    end
end
disp(['✅ 已生成动画：' gifName])
