'''
    Any-language Graphical Lemmatiser -xGL
    Copyright (C) 2025  Lulamile Mzamo - lula[underscore]mzamo[at]yahoo.co.uk

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU AFFERO GENERAL PUBLIC LICENSE as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.


    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from collections.abc import MutableMapping
from sklearn.semi_supervised import _self_training


class circTrie(MutableMapping):
    """
    A Circumfix tree.

    >>> mapping = dict(she=1, sells=5, sea=10, shells=19, today=5)
    >>> trie = Trie(mapping)
    >>> some_prefix = 'sh'
    >>> some_key = 'today'
    >>> print(*sorted(trie))
    sea sells she shells today
    >>> print(*sorted(trie.items()))
    ('sea', 10) ('sells', 5) ('she', 1) ('shells', 19) ('today', 5)
    >>> print(*trie.keys(prefix='sh'))
    she shells
    >>> trie['today'] = -55
    >>> trie['today']
    -55
    >>> len(trie)
    5
    >>> 'shells' in trie, 'shore' in trie
    (True, False)
    >>> del trie['shells']
    >>> len(trie)
    4
    """
    __slots__ =('_length','_root')
    # A dictionary with an optional :value: attribute.
#     class _ukEntry(object):
#         __slots__ = ('value')
#         def has_value(self):
#             return hasattr(self, 'value')
#     
    class _Entry(dict):
        __slots__ = ('value')
        def has_value(self):
            return hasattr(self, 'value') #self.value != None  #

    def __init__(self, *args, **kwargs):
        """
        Return a new Trie initialized from an optional positional
        argument and a possibly empty set of keyword arguments, as for
        the built-in type :dict:
        """
        self._root = self._Entry()
        self._length = 0
#         self.lastFind = None
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        """
        Gets the value corresponding to the specified key.
        :param key: an iterable of two iterables
        :return: the value corresponding to the key if it is in the Trie,
        otherwise raise KeyError.
        """
        entry = self._find(key)
        if type(entry) != self._Entry:
            return entry
        elif entry.has_value():
            return entry.value
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        """
        Sets or inserts the specified value, mapping the specified key to it
        :param key: an iterable to update or insert
        :param value: the object to associate with the key
        """
        entry = self._find(key, create=True)
        if not entry.has_value():
            self._length += 1
        entry.value = value

    def __delitem__(self, key):
        """
        Removes the value of the specified key from the Trie
        :param key: the iterable whose value is to be removed
        """
        entry = self._find(key)
        if entry.has_value():
            del entry.value
            self._length -= 1
        else:
            raise KeyError(key)

    def __len__(self):
        return self._length

    def __iter__(self):
        return iter(self.keys())

    def keys(self, circ=(), parents=False):
        """
        Searches for keys beginning with the specified circumfix : beginning with prefix and ending with suffix
        :param circ: an iterable
        :return: A generator yielding the keys one by one, as they are found
        """
        
        return (key for key, value in self.items(circ))

    def items(self, circ=()):
        """
        Searches for key value pairs where the keys begin with the specified
        prefix
        :param prefix: an iterable
        :return: A generator yielding tuples of keys and values as they are
        found
        """
        # prefix = tuple(circ[0]) +('|',)+ reversed(tuple(circ[1]))
        try:
            start = self._find(circ)
        except KeyError:
            raise StopIteration
        stack = [(circ, start)]
        while stack:
            current_prefix, entry = stack.pop()
            if entry.has_value():
                # print(current_prefix)
                splitLoc = current_prefix.index('|')
                circ = (current_prefix[:splitLoc],current_prefix[splitLoc+1:])
                yield (''.join(circ[0]),''.join(reversed(circ[1]))), entry.value
            for char, children in entry.items():
                stack.append((current_prefix + (char,), children))
    
#     def find_State(self,key):
#         key = tuple(key)
#         if len(self.currentState) ==0:
#             return ((),None)
#         elif key in self.currentState:
#             entry = self.currentState[key]
#             return (key,entry)
#         else:
#             keyGrams = affixList(key,reverse=True)
#             olap = keyGrams & set(self.currentState.keys())
#             
#             if len(olap) > 0: #There's a match:
#                 self.currentState = dict([(a,self.currentState[a]) for a in olap])
#                 cE = max(olap)
#                 cS = self.currentState[cE]
# 
#                 return (cE,cS)
#             else:
#                 return (0,None)
    
    def _find(self, key, create=False):
        # We assume the key is a tuple of two strings - a circumfix
#         if key == None :return self._root
#         current_path,entry = self.find_State(key)
        
#         if entry == None:
#         if self.lastFind and key[:len(self.lastFind[0])] == self.lastFind[0]:
#             entry =  self.lastFind[1]
#             key = key[len(self.lastFind[0]):]
#             path = self.lastFind[0]
#         else:
        entry = self._root
#             path = ()
#             current_path = ()
#             self.currentState.clear()
        #we store the key as a prefix + | reverse of suffix
        # print(key)
        if key: key = tuple(key[0]) +('|',)+ tuple(reversed(key[1]))
        
        for char in key: #[len(current_path):]:
#             path += (char,)
            if char in entry:
                
                entry = entry[char]
#                 self.lastFind = (path,entry)
#                 current_path = current_path + (char,)
#                 self.currentState[current_path]  = entry
            elif create:
#                 self.lastFind = (path[:-1],entry)
                new_entry = self._Entry()
                entry[char] = new_entry
                entry = new_entry
#                 current_path = current_path + (char,)
#                 self.currentState[current_path]  = entry
            else:
                raise KeyError(key)
        
        return entry    

    
    
    def __str__(self):
        return str({ (key, value) for key, value in self.items()})

def testCircTrie():
    import tracemalloc

    tracemalloc.start()
    mapping = ((('s','ea'),10), (('s','he'),(1,2)), (('sh','ells'),5),  (('she','lls'),19), (('to','day'),5))
    # mapping[('s','el')] = (45,)
#     mapping = [(tuple(key),value) for key,value in mapping.items()]
    trie = circTrie(mapping)
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
    some_prefix = tuple('sh')
    some_key = tuple('today')
    # print( sorted(trie))
    assert sorted(trie) == [('s', 'ea'), ('s', 'he'), ('sh', 'ells'), ('she', 'lls'), ('to', 'day')]
    assert sorted(trie.items()) == [(('s','ea'),10), (('s','he'),(1,2)), (('sh','ells'),5),  (('she','lls'),19), (('to','day'),5)]
    # assert [key for key in trie.keys(prefix=tuple('sh'))] == [tuple('she'), tuple('shells')]
    # trie['today'] = -55
    # assert trie['today'] == -55
    # assert len(trie) == 6
    # assert (tuple('shells') in trie, 'shore' in trie) == (True, False)
    # del trie['shells']
    # assert len(trie) ==     5
    # print([key for key in trie.keys(prefix=tuple('se'))])
    

def tests():
    testCircTrie()
    
if __name__ == "__main__":
    tests()
