function [FAMap, FA, pattern] = flipanglemap(FABase, size, pattern)
%%%% Date: %%%%
% 07/27/2021
%
%%%% Author information: %%%%
% Joseph Plummer
%
%%%% Agenda: %%%%
% A function to generate various flip angle maps (i.e. with different
% levels of smoothing and patterns, with different intensities).
% 
% Inputs:
% FABase = base flip angle to deviate flip angles from (e.g. mean FA).
% size = square image length of the FA map.
% pattern = name of desired function or pattern used to make flip angle map.
%
% Outputs:
% FAMap = flip angle map.
% FA = vector of the mean flip angles in each quadrant.
% pattern = name of function or pattern used to make flip angle map. 
% 
%% Regulate inputs:
if nargin == 3
    FABase = FABase;
    size = size;
    pattern = pattern;
elseif nargin == 2
    FABase = FABase;
    size = size;
    pattern = "quadrant"; % Default
else 
    disp(['The user must enter correct inputs for '...
        'flipanglemap(FABase, size, pattern), where FABase is the '...
        'approximate flip angle for the map to deviate from, size is the '...
        'flip angle map square length, and pattern is the function used '...
        'to make the flip angle map.'])
    return
end

% Ensure the inputs are correct:
inputs = {'Ones',...
    'Quadrant',...
    'Corner',...
    'Leftright',...
    'Trigonometry',...
    'Peaks',...
    'Cross',...
    'Windmill'}; 
if sum(strcmp(pattern, inputs)) == 1
    str = ["Flip angle map generated using the " + pattern + " function."];
    disp(str)
else
    disp('User must enter a suitable flip angle map function. Check function for list of suitable functions.')
    return
end


%% Generate different outputs:

% Set up 2D function inputs:
x = linspace(-0.5, 0.5, size);
y = linspace(-0.5, 0.5, size);
[X,Y] = meshgrid(x,y);

switch pattern
    case "Ones" 
        FAMap = [ones(size)*FABase]; % Constant flip angle
    case "Quadrant"
        FA = [FABase-10, FABase-5, FABase, FABase+5]; % Flip angle ranges
        FAMap = [ones(size/2)*FA(1) ones(size/2)*FA(2); ...
            ones(size/2)*FA(3) ones(size/2)*FA(4)];
    case "Corner"
        FAMap = FABase + (X + Y).*FABase; % Corner to corner
    case "Leftright"
        FAMap = FABase + FABase.*X.*exp(- X.^2 - Y.^2); % Left to right
    case "Trigonometry"
        FAMap = FABase + FABase.*(Y.*sin(X) - X.*cos(Y)); % Trigonometry
    case "Peaks"
        FAMap = FABase - 0.15*FABase.*peaks(size); % Peaks
    case "Cross"
        FAMap = FABase.*(0.5 - sign(sign((X.*10).^2 - 0.5 )-1 + sign((Y.*10).^2 - 1)-1)./2); % Cross
    case "Windmill"
        FAMap = FABase + FABase.*(sign(X.*Y) .* sign(1 - (X*10).^2 + (Y*10).^2)/4); % Windmill
    otherwise
end

% Deal with negatives/low-field areas:
FAMap(FAMap <= 4) = 4;

% Extract the mean flip angle for each quadrant:
FA = [mean(FAMap(1:size/2, 1:size/2),'all'),
    mean(FAMap(1:size/2, size/2 + 1:size),'all');
    mean(FAMap(size/2 + 1:size, 1:size/2),'all'),
    mean(FAMap(size/2 + 1:size, size/2 + 1:size),'all')];
FA = round(FA);

end