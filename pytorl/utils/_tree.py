import math
import operator


"""
this module contains useful tree data structures
"""


class _SegmentTree:
    def __init__(self, capacity, op, default_elem):
        assert capacity > 0 
        self.capacity = capacity
#         self._tree_len = 2 ** math.ceil(math.log(capacity, 2) + 1) - 1
        self._tree_len = 2 * capacity - 1
        self._op = op
        self._value = [default_elem for _ in range(self._tree_len)]
        self._idx_to_leaf = {}
        self._leaf_to_idx = {}
        self._init_mapping()
        
    def __len__(self):
        return self.capacity
        
    def inorder(self, idx=0):
        if idx >= self._tree_len: return
        self.inorder(2 * idx + 1)
        print(idx, self._value[idx])
        self.inorder(2 * idx + 2)
    
    def _init_mapping(self, idx=0):
        if 2 * idx + 1 >= self._tree_len: 
            self._idx_to_leaf[len(self._idx_to_leaf)] = idx
            self._leaf_to_idx[idx] = len(self._leaf_to_idx)
            return
        self._init_mapping(2 * idx + 1)
        self._init_mapping(2 * idx + 2)
    
    def _traverse_reduce(self, start, end, curr_node, node_start, node_end):
        if start == node_start and end == node_end: return self._value[curr_node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._traverse_reduce(start, end, 2 * curr_node + 1, node_start, mid)
        elif mid + 1 <= start:
            return self._traverse_reduce(start, end, 2 * curr_node + 2, mid + 1, node_end)
        else:
            return self._op(
                    self._traverse_reduce(start, mid, 2 * curr_node + 1, node_start, mid),
                    self._traverse_reduce(mid + 1, end, 2 * curr_node + 2, mid + 1, node_end)
                    )
            
    def reduce(self, start=0, end=None):
        if end is None: end = self.capacity
        if end < 0: end += self.capacity
        if end > self.capacity: raise ValueError('reduction out of capacity')
        # note: 'end' in this reduce operation is exclusive
        end -= 1
        return self._traverse_reduce(start, end, 0, 0, self.capacity - 1)
    
    
    def __setitem__(self, idx, val):
        # index of the leaf
        assert 0 <= idx < self.capacity, '__setitem__ index out of range'
        # get mapping
        idx = self._idx_to_leaf[idx]
        self._value[idx] = val
        while idx > 0:
            # get parent node
            idx = (idx - 1) // 2
            self._value[idx] = self._op(self._value[2 * idx + 1], self._value[2 * idx + 2])
            

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity, '__getitem__ index out of range'
        return self._value[self._idx_to_leaf[idx]]
    


class SumSegmentTree(_SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            op=operator.add,
            default_elem=0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super().reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5, 'prefixsum out of range'
        idx = 0
        while idx < self._tree_len - self.capacity:  # while non-leaf
            if self._value[2 * idx + 1] >= prefixsum:
                idx = 2 * idx + 1
            else:
                prefixsum -= self._value[2 * idx + 1]
                idx = 2 * idx + 2
        return self._leaf_to_idx[idx]


class MinSegmentTree(_SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            op=min,
            default_elem=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super().reduce(start, end)