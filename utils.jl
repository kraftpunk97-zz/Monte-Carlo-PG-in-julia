using Plots: plot, savefig, xlabel!, ylabel!

function plotresults(mean_rewards, eps_rewards)
    # Plotting rewards per episode
    x_axis = 1:length(mean_rewards)
    @assert length(eps_rewards) == length(mean_rewards) # Just to be safe...
    y_axis = hcat(eps_rewards, mean_rewards)
    p = plot(x_axis, y_axis,
        title="Monte Carlo Policy Gradient Method on CartPole-v0",
        label=["episodic reward", "10-episode average of reward"],
        lw=2, legend=:topleft)
    xlabel!("Episode #")
    ylabel!("Rewards")
    savefig(p, "Reward.png")
end
