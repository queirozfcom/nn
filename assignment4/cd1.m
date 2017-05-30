function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
     
    visible_data = sample_bernoulli(visible_data); 
     
    hidden_probs = visible_state_to_hidden_probabilities(rbm_w,visible_data);
    
    sampled_hidden_states = sample_bernoulli(hidden_probs);
    
    g1 = configuration_goodness_gradient(visible_data,sampled_hidden_states);
                
    reconstructed_visible_probs = hidden_state_to_visible_probabilities(rbm_w,sampled_hidden_states);
    
    reconstructed_visible_states = sample_bernoulli(reconstructed_visible_probs);
      
    reconstructed_hidden_probs = visible_state_to_hidden_probabilities(rbm_w,reconstructed_visible_states);
    
    % reconstructed_hidden_state = sample_bernoulli(reconstructed_hidden_probs);
    
    g2 = configuration_goodness_gradient(reconstructed_visible_states,reconstructed_hidden_probs);
       
    ret = g1-g2;
    
end
