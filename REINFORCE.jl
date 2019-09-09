using Flux
using Flux.Tracker
using Flux.Optimise: ADAM
using Gym: reset!, make, step!, render!
using DataStructures: CircularBuffer

#-----------------Environment Setup--------------------------------------------#
env = make("CartPole-v0", :human_pane)
reset!(env)
ACTION_SPACE = 2
OBSERVATION_SPACE = 4
REWARD_THRESHOLD = 200
MAX_EPISODES = 5000

#---------------------Model Architechture--------------------------------------#
model = Chain(
    Dense(OBSERVATION_SPACE, 128, relu),
    Dense(128, ACTION_SPACE),
    softmax
)
opt = ADAM(3f-4)
const γ = 9f-1

#W1, b1, W2, b2 = params(model).order

#-------------------- Functions -----------------------------------------------#

include("forwardpass.jl")
include("backwardpass.jl")
include("utils.jl")

function episode()
    current_state = reset!(env)
    rewards = Float32[]
    log_probs = []

    terminal = false

    while !terminal
        action, log_prob = execpolicy(model, current_state)
        next_state, reward, terminal, _ = step!(env, action)

        push!(rewards, reward)
        push!(log_probs, log_prob)

        current_state = next_state
    end
    return rewards, log_probs
end

function learnpolicy()
    eps_rewards = Float32[]
    mean_rewards = Float32[]

    for eps=1:MAX_EPISODES
        rewards, log_probs = episode()
        policyupdate!(rewards, log_probs)
        eps_reward = sum(rewards)
        push!(eps_rewards, eps_reward)


        mean_reward = eps < 10 ? mean(eps_reward) : mean(eps_rewards[end-9:end])
        push!(mean_rewards, mean_reward)
        println("Episode $eps = $eps_reward | Mean of the last 10 episodes = $mean_reward")
        mean_reward ≥ REWARD_THRESHOLD && break
    end

    plotresults(mean_rewards, eps_rewards)
end

function policydemo()
    current_state = reset!(env)
    terminal = false
    for frame_num ∈ 1:500
        action, _ = execpolicy(model, current_state)
        next_state, reward, terminal, _ = step!(env, action)

        render!(env)

        terminal && (next_state = reset!(env))

        current_state = next_state
        sleep(0.001)
    end
end

# Driver code
function REINFORCE()
    learnpolicy()
    policydemo()
end


REINFORCE()
