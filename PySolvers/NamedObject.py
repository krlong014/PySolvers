
class NamedObject:
    '''NamedObject is a base class for objects with names.'''

    def __init__(self, name=''):
        '''Constructor.'''
        self._name = name

    def name(self):
        '''Return the name of the object.'''
        return self._name
