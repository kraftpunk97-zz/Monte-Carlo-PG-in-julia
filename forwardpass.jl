using Distributions: ProbabilityWeights, sample

function execpolicy(model, current_state)
    probs = model(current_state)
    action = sample(1:2, ProbabilityWeights(probs))
    log_prob = log(probs[action])
    return action, log_prob
end

#=
function getaction(current_state)
    x = relu.(W1 * current_state .+ b1)
    probs = softmax(W2 * x .+ b2)
    action = argmax(probs)
    log_probab = log(probs[action])
    return action, log_probab
end
=#
