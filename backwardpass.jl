using Flux.Optimise: update!
using Statistics: std, mean

function policyupdate!(rewards, log_probs)
    discounted_rewards = Float32[calcreward(rewards[t:end]) for t=1:length(rewards)]

    # Using normalized discounted rewards, because Andrej Karpathy said so.
    # This is a hack to control the variance of the policy gradient estimator.
    norm_discounted_rewards = (discounted_rewards .- mean(discounted_rewards)) ./ (std(discounted_rewards) .+ 1f-6)

    policy_grads = -log_probs .* norm_discounted_rewards

    ps = params(model)
    gs = gradient(()->sum(policy_grads), ps)

    update!(opt, ps, gs)
end

function calcreward(rewards_subset)
    γ_subarray = ones(Float32, length(rewards_subset)-1)
    γ_subarray *= γ
    γ_array = cumprod(vcat(Float32[1], γ_subarray))
    discouonted_goal = sum(γ_array .* rewards_subset)
    return discouonted_goal
end
