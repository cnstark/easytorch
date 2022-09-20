# Modified from: https://github.com/makinacorpus/easydict/blob/master/easydict/__init__.py

class Config(dict):
    """
    Get attributes

    >>> d = EasyDict({'foo':3})
    >>> d['foo']
    3
    >>> d.foo
    3
    >>> d.bar
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'bar'

    Works recursively

    >>> d = EasyDict({'foo':3, 'bar':{'x':1, 'y':2}})
    >>> isinstance(d.bar, dict)
    True
    >>> d.bar.x
    1

    Bullet-proof

    >>> EasyDict({})
    {}
    >>> EasyDict(d={})
    {}
    >>> EasyDict(None)
    {}
    >>> d = {'a': 1}
    >>> EasyDict(**d)
    {'a': 1}

    Set attributes

    >>> d = EasyDict()
    >>> d.foo = 3
    >>> d.foo
    3
    >>> d.bar = {'prop': 'value'}
    >>> d.bar.prop
    'value'
    >>> d
    {'foo': 3, 'bar': {'prop': 'value'}}
    >>> d.bar.prop = 'newer'
    >>> d.bar.prop
    'newer'


    Values extraction

    >>> d = EasyDict({'foo':0, 'bar':[{'x':1, 'y':2}, {'x':3, 'y':4}]})
    >>> isinstance(d.bar, list)
    True
    >>> from operator import attrgetter
    >>> map(attrgetter('x'), d.bar)
    [1, 3]
    >>> map(attrgetter('y'), d.bar)
    [2, 4]
    >>> d = EasyDict()
    >>> d.keys()
    []
    >>> d = EasyDict(foo=3, bar=dict(x=1, y=2))
    >>> d.foo
    3
    >>> d.bar.x
    1

    Still like a dict though

    >>> o = EasyDict({'clean':True})
    >>> o.items()
    [('clean', True)]

    And like a class

    >>> class Flower(EasyDict):
    ...     power = 1
    ...
    >>> f = Flower()
    >>> f.power
    1
    >>> f = Flower({'height': 12})
    >>> f.height
    12
    >>> f['power']
    1
    >>> sorted(f.keys())
    ['height', 'power']

    update and pop items
    >>> d = EasyDict(a=1, b='2')
    >>> e = EasyDict(c=3.0, a=9.0)
    >>> d.update(e)
    >>> d.c
    3.0
    >>> d['c']
    3.0
    >>> d.get('c')
    3.0
    >>> d.update(a=4, b=4)
    >>> d.b
    4
    >>> d.pop('a')
    4
    >>> d.a
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'a'
    """

    # pylint: disable=super-init-not-called
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__:
            if not (k.startswith('__') and k.endswith('__')) and not k in ('get', 'update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            v = [self.__class__(x) if isinstance(x, dict) else x for x in value]
            # Don't repalce tuple with list
            if isinstance(value, tuple):
                v = tuple(v)
            value = v
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, key):
        # Support `cfg[AA.BB.CC]`
        if isinstance(key, str):
            keys = key.split('.')
        else:
            keys = key
        value = super().__getitem__(keys[0])
        if len(keys) > 1:
            return value.__getitem__(keys[1:])
        else:
            return value

    def has(self, key):
        return self.get(key) is not None

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, e=None, **f):
        d = e or {}
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        # Check for existence
        if hasattr(self, k):
            delattr(self, k)
        return super().pop(k, d)