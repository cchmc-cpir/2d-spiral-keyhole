function [undersamplefigure] = undersamplefunction(us_r0, us_r, us_0, us_1, utype)
%%%% Date: %%%%
% 08/04/2021
%
%%%% Author information: %%%%
% Joseph Plummer
%
%%%% Agenda: %%%%
% A function to show how the spiral density varies over normalized k-space
% radius, r. Theory behind these functions can be found in slides by Joseph
% Plummer, which are backed up in part by the Philips BNI spiral coords
% code.
%
% Inputs: Arguments of a piecewise undersample function, R(r).
% us_r0 = start undersample factor (0.25 - 2)
% us_r = end undersample factor (0.25 - 2)
% us_0 = start function position (0 - 1)
% us_1 = end function position (0 - 1)
% utype = undersampling type (none, linear, quadratic, hanning)
%
% Outputs:
% undersamplefigure = figure
%
%% Script:
% Input define:
if nargin == 5
    disp(["User entered us_r0 = ",num2str(us_r0)])
    disp(["User entered us_r = ",num2str(us_r)])
    disp(["User entered us_0 = ",num2str(us_0)])
    disp(["User entered us_1 = ",num2str(us_1)])
    disp(["User entered utype = ",num2str(utype)])
    
else
    disp("User must define values for the start and end undersample factors, the start and end function locations, and the variable density function type.")
    return
end

%% Produce a function and plot for each undersample variable density type:
% Various undersample function types:
switch utype
    case 0 % Linear (can also be uniform if us_r = us_r0)
        syms R(r)
        R(r) = piecewise(0 <= r < us_0, us_r0,...
            us_0 <= r < us_1, us_r0 + (us_r - us_r0).*((r - us_0)./(us_1 - us_0)),...
            us_1 <= r <= 1, us_r)
        type = "Linear";
    case 1 % Quadratic
        syms R(r)
        R(r) = piecewise(0 <= r < us_0, us_r0,...
            us_0 <= r < us_1, us_r0 + (us_r - us_r0).*((r - us_0)./(us_1 - us_0)).^2,...
            us_1 <= r <= 1, us_r)
        type = "Quadratic";
    case 2 % Hanning
        syms R(r)
        R(r) = piecewise(0 <= r < us_0, us_r0,...
            us_0 <= r < us_1, us_r0 + (us_r - us_r0).*sin(pi*((r - us_0)./(us_1 - us_0))./2).^2,...
            us_1 <= r <= 1, us_r)
        type = "Hanning";
    otherwise
        disp("Undersample type must be one of the options: 0, 1, 2.")
        return
end

% Plot the results:
undersamplefigure = figure('Name','Radial undersampling profile')
hold on
fplot(R, 'r--','LineWidth', 2)
title('2D spiral radial undersampling profile','FontSize',14)
subtitle(["Start undersample factor: R_{s} = " + num2str(us_r0);
    "End undersample factor: R_{e} = " + num2str(us_r);
    "Function start: R_{0} = " + num2str(us_0);
    "Function end: R_{1} = " + num2str(us_1);
    "Undersample function type: " + type])
xlabel('Normalized radius \it{r} (fraction of k_{max})','FontSize',14)
ylabel('Undersampling factor function \it{R(r)}','FontSize',14)
ylim([0 2])
xlim([0 1])
grid on

end

