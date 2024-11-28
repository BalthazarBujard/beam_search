from typing import Any, List, Callable, Dict, Tuple, Optional
import torch
import numpy as np
import copy

class Candidate():
    def __init__(self,states:List[Any], probs:List[float], terminal_state : Optional[int] = None):
        self.states = states #(T,)
        self.probs = probs #(T,)
        self.terminal_state = terminal_state
        self.effective_length = len(self.states) #length of sequence until terminal state is found
        self.terminated = False #True if terminal state is assigned to candidate sequence
        
    def update(self,state:Any,prob:float):
        #update states and probs
        self.states+=[state]
        self.probs+=[prob]
        
        #update/check effective length and terminated state
        if not self.terminated :
            self.effective_length = len(self.states)
            if state == self.terminal_state:
                self.terminated = True
    
    def compute_prob(self) -> float:
        return np.prod(self.probs[:self.effective_length]) #prod(p(yt|y<t)) until y_t == terminal state
    
    def compute_score(self) -> float:
        return sum(np.log(self.probs[:self.effective_length]))/(self.effective_length**0.75)
    
    @property
    def score(self):
        return self.compute_score()
    
    def __str__(self):
        return f"states : {self.states}\nscore : {self.score}"
    
    
class BeamSearch():
    def __init__(self, transition_fn : Callable, transition_fn_args : Optional[Dict], terminal_state : Optional[int] = None):
        
        self.transition_fn = transition_fn #function to compute probabilities over
        self.transition_fn_args = transition_fn_args #additional arguments for transition function
        self.terminal_state = terminal_state #equivalent of End Of Sentence token
    
    def __call__(self, x_init : torch.Tensor, beam_width : int, max_len : int) -> List[Candidate]:
        
        B, L0 = x_init.shape #x_init can have an initial state greater than 1 (?)
        
        #init
        #list of (state,score) where state is the sequence of idx and score the total conditional probability = prod p(y_t|y<t)
        candidates =[[Candidate([x_init[b].item()], [1], self.terminal_state) for _ in range(beam_width)] for b in range(B)] #(B,beam_width)
        
        for _ in range(max_len):
            candidates = self.__search_one_step(candidates)

        best_candidates = [sorted(candidates_group,key=lambda x : x.score, reverse=True)[0] for candidates_group in candidates] #(B,)
        
        return best_candidates
    
    def __search_one_step(self, candidates : List[List[Candidate]]):
        
        beam_width = len(candidates[0])
        
        probs = self.transition_fn(candidates,**self.transition_fn_args) #(B,beam_width,state space size)
        
        for batch_index, beam_probs in enumerate(probs) :
            #print("-*-*-*-")
            #beam_probs (beam_width, state space size)
            this_candidates = candidates[batch_index]
            
            # WE WANT TO MAXIMIZE THE prod(P(Y_t|Y<t)) so before doing find_k_best we need to multiply the prob by the score of the candidate
            #get probabilities (prob of prod(P(y_t-1|y<t-1)))
            
            # TODO : TERMINATED CANDIDATES CAN BE KEPT IF THEIR SCORE IS GREATER THAN OTHER CANDIDATES OTPIONS 
            # BUT WE NEED TO COMPARE THE NEW BEAM_STATES_LIKELIHOOD (OF NON-TERMINATED SEQUENCES) WITH THE TERMINATED ONES
            
            candidates_probs = torch.tensor([c.compute_prob() for c in this_candidates], device=beam_probs.device).view(-1,1) #(beam_width,1)
            #update beam_probs
            beam_states_log_likelihood = torch.log(beam_probs*candidates_probs)
            
            topk_states= self.__find_k_best_states(beam_states_log_likelihood, beam_width) #(beam_width, 2)
            
            # retrieve state probability from beam_probs
            topk_probs = beam_probs[topk_states[:,0],topk_states[:,1]].numpy(force=True)
            
            
            #update candidates with new states
            updated_candidates = self.__update_candidates(this_candidates, topk_states, topk_probs)
            
            #assign updated candidates
            candidates[batch_index] =  updated_candidates    
        
        return candidates
    
    def __update_candidates(self, candidates : List[Candidate], topk_states : np.ndarray, topk_probs : np.ndarray):
        
        #updated_candidates = [[None] for _ in range(len(candidates))]
        candidates_to_update : List[Candidate] = [copy.deepcopy(candidates[idx]) for idx in topk_states[:,0]] #retrieve best candidates to continue
            
        for k,(new_state,prob) in enumerate(zip(topk_states[:,1],topk_probs)):
            #print("candidate index:",k)
             
            #candidate_to_continue = candidates[index[0]]
            #print(f"prev candidate:\n{prev_candidate}")
            #new_state = index[1] 
            
            candidates_to_update[k].update(new_state, prob)
            
            # state_score = np.log(prob)
            
            # new_states = candidate_to_continue.states + [new_state]
            # #new_states_vectors = torch.cat([prev_candidate.states_vectors, token_vector.view(1,1,-1)],dim=1)
            # new_score = candidate_to_continue.score + state_score
            
            # updated_candidates[k]=Candidate(new_states, new_score)
        
        return candidates_to_update
    
    def __find_k_best_states(self, probs : torch.Tensor, beam_width : int) -> np.ndarray:
        #probs : (beam_width, state space size)
        
        topk_states_flat = torch.topk(probs.flatten(),beam_width)[1]
        
        #topk_states is an array of indexes corresponding to the best previous states and the new state
        #that way we can continue the candidates that maximizes score and forget non-maximum candidates
        topk_states = np.unravel_index(topk_states_flat.numpy(force=True),probs.shape)
        topk_states = np.column_stack(topk_states) #convert to (beam_width,3) format. 3 is if we work with batched inputs and 2 if unbatched (equal to len(shape))
        
        
        return topk_states
