from typing import Any, List, Callable, Dict, Tuple, Optional
import torch
import numpy as np
import copy

#TODO : CHANGE FROM SCORE TO SCORES AS LIST[FLOAT] AND COMPUTE SCORE WITH THAT LIST AND THE ACTUAL LENGTH OF THE CANDIDATE
class Candidate():
    def __init__(self,states:List[Any], probs:List[float]):
        self.states = states #(T,)
        self.probs = probs #(T,)
        
    def update(self,state:Any,prob:float):
        self.states+=[state]
        self.probs+=[prob]
    
    def compute_prob(self):
        return np.prod(self.probs) #log(prod(p(yt|y<t)))
    
    #TODO : PRENDRE EN COMPTE TAILLE EFFECTIVE DE SEQUENCE PREDITE
    def compute_score(self):
        return sum(np.log(self.probs))
    
    @property
    def score(self):
        #raise NotImplementedError()
        return self.compute_score()
    
    def __str__(self):
        return f"states : {self.states}\nscore : {self.score}"
    
    
class BeamSearch():
    def __init__(self, transition_fn : Callable, transition_fn_args : Optional[Dict]):
        
        self.transition_fn = transition_fn #function to compute probabilities over
        self.transition_fn_args = transition_fn_args #additional arguments for transition function
    
    def __call__(self, x_init : torch.Tensor, beam_width : int, max_len : int) -> List[Candidate]:
        
        B, L0 = x_init.shape #x_init can have an initial state greater than 1 (?)
        
        #init
        #list of (state,score) where state is the sequence of idx and score the total conditional probability = prod p(y_t|y<t)
        candidates =[[Candidate([x_init[b].item()], [1]) for _ in range(beam_width)] for b in range(B)] #(B,beam_width)
        
        for _ in range(max_len):
            candidates = self.__search_one_step(candidates)

        best_candidates = [sorted(candidates_group,key=lambda x : x.score, reverse=True)[0] for candidates_group in candidates] #(B,)
        
        return best_candidates
    
    def __search_one_step(self, candidates: List[List[Candidate]]):
        
        beam_width = len(candidates[0])
        
        probs = self.transition_fn(candidates,**self.transition_fn_args) #(B,beam_width,state space size)
        
        # ATTENTION : 
        # WE WANT TO MAXIMIZE THE prod(P(Y_t|Y<t)) so before doing find_k_best we need to multiply the prob by the score of the candidate
        for batch_index, beam_probs in enumerate(probs) :
            #print("-*-*-*-")
            #beam_probs (beam_width, state space size)
            this_candidates = candidates[batch_index]
            
            #get probabilities (prob of prod(P(y_t-1|y<t-1)))
            candidates_probs = torch.tensor([c.compute_prob() for c in this_candidates], device=beam_probs.device).view(-1,1) #(beam_width,1)
            #update beam_probs
            beam_states_likelihood = beam_probs*candidates_probs 
            
            topk_states= self.__find_k_best_states(beam_states_likelihood, beam_width) #(beam_width, 2)
            
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
