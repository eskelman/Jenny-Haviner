# Plot the Reynolds Stress and Expansion coefficient curves for the results
using DataFrames,CSV

#DNS numbers (specifically Re180)
DNS_results_location = joinpath(pwd(), "Code", "DNS", "results")
DNS_results = CSV.read(joinpath(DNS_results_location, "train_out_Re180.csv"), DataFrame)
results_plot = plot(DNS_results.y_plus, DNS_results.var_uu, label = "DNS @ Re_τ = 180", lc=:red)

#Non-Normalised
#Results from machine learning
results_location = joinpath(pwd(), "Code", "Post Processing", "results")
predicted_results = CSV.read(joinpath(results_location, "RS_results_non_normalised.csv"), DataFrame)

#Plot the DNS and ML results
scatter!(predicted_results.y_plus, predicted_results.uu_yhat, xlim = (0,100),ylim = (0,10), label="All ML Predicitons (non-norm)", mc=:black, markershape=:cross, ms=1)

#Highlight the Re180 ML results
predicted_results_180 = CSV.read(joinpath(results_location, "Re180.csv"), DataFrame)
plot!(predicted_results_180.y_plus, predicted_results_180.uu_yhat, xlim = (0,100),ylim = (0,10), label="ML Prediction (non_norm)", lc=:blue)

#Normalised
#Results from machine learning
results_location = joinpath(pwd(), "Code", "Post Processing", "results")
predicted_results = CSV.read(joinpath(results_location, "RS_results_normalised.csv"), DataFrame)

#Plot the DNS and ML results
scatter!(predicted_results.y_plus, predicted_results.uu_yhat, xlim = (0,100),ylim = (0,10), label="All ML Predicitons (norm)", shape=:hline, mc=:black, ms=1)

#Highlight the Re180 ML results
predicted_results_180 = CSV.read(joinpath(results_location, "Re180_norm.csv"), DataFrame)
plot!(predicted_results_180.y_plus, predicted_results_180.uu_yhat, xlim = (0,100),ylim = (0,10), label="ML Prediction (norm)", lc=:purple)