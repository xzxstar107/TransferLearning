# 4. Neural network components and models

# Define a gradient reversal layer used to negate gradients during backpropagation
class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

# Define a neural network module for estimating demand
class Demand(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(Demand, self).__init__()
        # Create a list of layers based on the input, output, and hidden layers
        layers = [nn.Linear(input_dim, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        # Register the sequence of layers as a module
        self.layers = nn.Sequential(*layers)

    def forward(self, p, x):
        # Concatenate price 'p' and other features 'x'
        concatenated_input = torch.cat([p, x], dim=-1)
        # Pass the concatenated input through the network
        return self.layers(concatenated_input)

# Define the main model class combining multiple neural network components
class Model(nn.Module):
    def __init__(self, demand_net):
        super(Model, self).__init__()
        self.demand_net = demand_net
        # Define other components such as price function and critic (optional)
        # self.price_net = ...
        # self.critic_net = ...

    def forward(self, features):
        # Extract price and other features
        price = features[:, 0].unsqueeze(1)
        other_features = features[:, 1:]
        # Compute demand
        demand_estimate = torch.sigmoid(self.demand_net(price, other_features))
        # Compute other outputs if necessary
        # price_estimate = self.price_net(other_features)
        # domain_classification = self.critic_net(other_features)
        # return all outputs
        return demand_estimate

# Instantiate the demand estimation network
input_dim = 1
output_dim = 1
hidden_layers = [15, 15]
demand_net = Demand(input_dim, output_dim, hidden_layers)

# Instantiate the main model with the demand network
model = Model(demand_net)

# Optionally, apply custom initializations to the model
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
