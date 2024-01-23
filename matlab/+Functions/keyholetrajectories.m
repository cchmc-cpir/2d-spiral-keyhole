function [traj_key1,traj_key2,keyhole1,keyhole2] = keyholetrajectories(traj,traj_rad,keyhole_rad,time,narms_Key1,narms_Key2)
%%%% Date: %%%%
% 07/15/2021
%
%%%% Author information: %%%%
% Joseph Plummer
%
%%%% Agenda: %%%%
% A function to generate a keyhole animation for the 2-key 2D-spiral approach.
% This is designed to show the simultaneous spiral trajectories for each RF
% pulse.
% 
% Inputs:
% traj = trajectory coordinates
% traj_rad = trajectory radius for each coordinate (use pythagoras)
% keyhole_rad = keyhole radius
% time = desired pause amount for animation
% narms_Key1 = number of arms in key 1
% narms_Key2 = number of arms in Key 2
%
% Outputs:
% traj_key = coordinates for each trajectory point in each key
% keyhole = coordinates for each trajectory point in the keyhole
% 
%% Script:
% Input define:
if nargin <= 4
    narms_Key1 = floor(size(traj,3)/2);
    narms_Key2 = size(traj,3) - narms_Key1;
else
end

% General procedure:
% Generate keys: (first temporal half go to key 1, second temporal half go to key 2)
% traj_key1 = zeros(size(traj,1), size(traj,2), size(traj,3));
% traj_key2 = zeros(size(traj,1), size(traj,2), size(traj,3));

% Key 1:
for i = 1:narms_Key1 % Number of spirals in key 1.
    traj_key1(:,:,i) = traj(:,(traj_rad(:,i) < keyhole_rad),i);
    keyhole1(:,:,i) = traj(:,(traj_rad(:,i) > keyhole_rad),i);
end

% Key 2:
for i = 1:narms_Key2 % Number of spirals in key 2.
    traj_key2(:,:,i) = traj(:,(traj_rad(:,i + narms_Key1) < keyhole_rad),i + narms_Key1);
    keyhole2(:,:,i) = traj(:,(traj_rad(:,i + narms_Key1) > keyhole_rad),i + narms_Key1);
end

%% Plot features:
lw = 0.3;
% Key 1:
for i = 1:narms_Key1 
    hold on
    plot3(squeeze(traj_key1(1,:,i)),squeeze(traj_key1(2,:,i)),...
        i*ones(size(traj_key1,2),1),'m-','LineWidth',lw)
    plot3(squeeze(traj_key1(1,end,i)),squeeze(traj_key1(2,end,i)),i,'yo','MarkerFaceColor','yellow')
    plot3(squeeze(keyhole1(1,:,i)),squeeze(keyhole1(2,:,i)),...
        i*ones(size(keyhole1,2),1),'b-','LineWidth',lw)
    plot3(squeeze(keyhole1(1,end,i)),squeeze(keyhole1(2,end,i)),i,'ro','MarkerFaceColor','red')
    xlabel('k_{x}')
    ylabel('k_{y}')
    zlabel('Projection number')
    xlim([-0.5 0.5]);
    ylim([-0.5 0.5]);
    zlim([0 1*size(traj,3)]);
    view(20, 20)
    grid on
    ax = gca;
    c = ax.Color;
    ax.Color = 'black';
    title('Keyhole Visualization')
    pause(time)

end
% Add key 2:
for i = 1:narms_Key2
    hold on
    plot3(squeeze(traj_key2(1,:,i)),squeeze(traj_key2(2,:,i)),...
        (i+narms_Key1)*ones(size(traj_key2,2),1),'c-','LineWidth',lw)
    plot3(squeeze(traj_key2(1,end,i)),squeeze(traj_key2(2,end,i)),(i+floor(size(traj,3)/2)),'yo','MarkerFaceColor','yellow')
    plot3(squeeze(keyhole2(1,:,i)),squeeze(keyhole2(2,:,i)),...
        (i+narms_Key1)*ones(size(keyhole2,2),1),'b-','LineWidth',lw)
    plot3(squeeze(keyhole2(1,end,i)),squeeze(keyhole2(2,end,i)),(i+floor(size(traj,3)/2)),'ro','MarkerFaceColor','red')
    pause(time)
end

hold off
end

