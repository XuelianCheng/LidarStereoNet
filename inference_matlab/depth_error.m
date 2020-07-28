function [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3] = depth_error(gt,pred)
    
    thresh = max((gt ./ pred), (pred ./ gt));
    a1 = mean(thresh < 1.25   );
    a2 = mean(thresh < 1.25 .^2);
    a3 = mean(thresh < 1.25 .^3);
    
    mae = mean(abs(gt-pred));
    rmse = (gt - pred) .^ 2;
    rmse = sqrt(mean(rmse));

    rmse_log = (log(gt) - log(pred)) .^2;
    rmse_log = sqrt(mean(rmse_log));

    abs_rel = mean(abs(gt - pred) ./ gt);
    sq_rel = mean(((gt - pred).^2) ./ gt);
    



