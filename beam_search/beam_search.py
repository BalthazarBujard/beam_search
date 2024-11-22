from typing import Any, List, Callable, Dict, Tuple
import torch
import numpy as np

class Candidate():
    def __init__(self,states:List[Any], score:float):
        self.states = states #(T,)
        self.score = score
        
    def update(self,state:Any,score:float):
        self.states+=[state]
        self.score+=score
    
    def __str__(self):
        return f"states : {self.states}\nscore : {self.score}"
    
    
class BeamSearch():
    def __init__(self, transition_fn : Callable, transition_fn_args : Dict):
        self.transition_fn = transition_fn
        self.transition_fn_args = transition_fn_args
    
    def __call__(self, x_init : torch.Tensor, beam_width : int, max_len : int) -> List[Candidate]:
        
        B, L0 = x_init.shape #x_init can have an initial state greater than 1 (?)
        
        #init
        #list of (state,score) where state is the sequence of idx and score the total conditional probability = prod p(y_t|y<t)
        candidates =[[Candidate([x_init[b].item()], 0) for _ in range(beam_width)] for b in range(B)] #(B,beam_width)
        
        for _ in range(max_len):
            #print("----------")
            #probs=torch.full(np.shape(candidates)+(vocab_size,),fill_value=-torch.inf) #init probs as -inf
            
            candidates = self.__search_one_step(candidates)


        best_candidates = [sorted(candidates_group,key=lambda x : x.score, reverse=True)[0] for candidates_group in candidates]
        
        return best_candidates
    
    def __search_one_step(self, candidates: List[List[Candidate]]):
        
        beam_width = len(candidates[0])
        
        probs = self.transition_fn(candidates,**self.transition_fn_args) #(B,beam_width,vocab_size)
            
        #print("probs shape:",probs.shape)
        
        for batch_index, beam_probs in enumerate(probs) :
            #print("-*-*-*-")
            #beam_probs (beam_width,vocab_size)
            topk_states, topk_probs = self.__find_k_best_states(beam_probs, beam_width) #(beam_width, 2), (beam_width)
            
            #update candidates with new states
            updated_candidates = self.__update_candidates(candidates[batch_index], topk_states, topk_probs)
            
            #assign updated candidates
            candidates[batch_index] =  updated_candidates
            
        
        return candidates
    
    def __update_candidates(self, candidates : List[Candidate], topk_states : np.ndarray, topk_probs : np.ndarray):
        
        updated_candidates = [[None] for _ in range(len(candidates))]
            
        for k,(index,prob) in enumerate(zip(topk_states,topk_probs)):
            #print("candidate index:",k)
             
            candidate_to_continue = candidates[index[0]]
            #print(f"prev candidate:\n{prev_candidate}")
            new_state = index[1] 
            
            state_score = np.log(prob)
            
            new_states = candidate_to_continue.states + [new_state]
            #new_states_vectors = torch.cat([prev_candidate.states_vectors, token_vector.view(1,1,-1)],dim=1)
            new_score = candidate_to_continue.score + state_score
            
            updated_candidates[k]=Candidate(new_states, new_score)
        
        return updated_candidates
    
    def __find_k_best_states(self, probs : torch.Tensor, beam_width : int) -> Tuple[np.ndarray, np.ndarray]:
        
        topk_probs, topk_states_flat = torch.topk(probs.flatten(),beam_width) 
        
        #topk_states is an array of indexes corresponding to the best previous states and the new state
        #that way we can continue the candidates that maximizes score and forget non-maximum candidates
        topk_states = np.unravel_index(topk_states_flat.numpy(force=True),probs.shape)
        topk_states = np.column_stack(topk_states) #convert to (beam_width,3) format. 3 is if we work with batched inputs and 2 if unbatched (equal to len(shape))
        
        
        return topk_states, topk_probs.numpy(force=True)
