import os
import time
import re
import numpy as np

#para correr remotamente
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image as pil_image


def bitificar8(im,desv=1):
    '''Devuelve la imagen pasada a 8 bits. Puede reescalar la desviación.'''
    #normalizo la imagen            
    im -= im.mean()
    im /= (im.std() + 1e-5) #por si tengo cero
    im *= desv
    #la llevo a 8 bits
    im *= 64
    im += 128
    im = np.clip(im, 0, 255).astype('uint8') #corta todo por fuera de la escala
    return im

def new_name(name, newformater='_%d'):
    '''
    Returns a name of a unique file or directory so as to not overwrite.
    
    If proposed name existed, will return name + newformater%number.
     
    Parameters:
    -----------
        name : str (path)
            proposed file or directory name influding file extension
        newformater : str
            format to give the index to the new name, esuring a unique name
            
    Returns:
    --------
        name : str
            unique namefile using input 'name' as template
    '''
    
    #if file is a directory, extension will be empty
    base, extension = os.path.splitext(name)
    i = 2
    while os.path.exists(name):
        name = base + newformater%i + extension
        i += 1
        
    return name

def make_dirs_noreplace(dirs_paths):
    """
    Creates a new directory only if that directory doesn't exist already'

    Parameters
    ----------
    dirs_paths : str, path-like object
        The candidate directory to create

    Returns
    -------
    dirs_paths : str, path-like object
        same string or path object

    """
    
    try:
        os.makedirs(dirs_paths)
    except FileExistsError:
        print('While creating ', dirs_paths, 'found it already exists.')

    return dirs_paths

def natural_sort(l): 
    """
    Applies natural sort to a list, returns new list. Natural sort treats 
    strings of numbers as one unit, in contrast to alphanumerical (or 
    lexicographic) ordering. This means that 11 comes before 2 in alphanimeric
    sorting, but 2 before 11 in natural sort.

    Parameters
    ----------
    l : iterable
        iterable to sort

    Returns
    -------
    list
        sorted list from the items in l, naturally sorted
        
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def find_numbers(string):
    
    """Returns a list of numbers found on a given string
    
    Parameters
    ----------
    string: str
        The string where you search.
    
    Returns
    -------
    list
        A list of numbers (each an int or float).
    
    Raises
    ------
    "There's no number in this string" : TypeError
        If no number is found.
    """
    
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", string)
    
    if not numbers:
        raise TypeError("There's no number in this string")
    
    for i, n in enumerate(numbers):
        if '.' in n:
            numbers[i] = float(n)
        else:
            numbers[i] = int(n) 
    
    return numbers


def iter_to_csv(iterable, fmt='.2f'):
    """
    Joins an iterable into a string of comma separated values.

    Parameters
    ----------
    iterable : iterable
        An iterable to join into comma separated values.
    fmt : TYPE, optional
        A format string to be used when formating the values of the iterable. 
        The default is '.2f'.

    Returns
    -------
    str
        A string of the values in the iterable, separated by commas.

    """
    fmt_string = f'{{:{fmt}}}'
    return ','.join(map(lambda s: fmt_string.format(s), iterable))

class contenidos(list):
    '''Subclase de lista que toma un directorio y crea una lista de los contenidos.
    
    -----------
    Parámetros:
        carpeta : str
            Directorio cuyo contenido se quiere recolectar. Default: el directorio de trabajo actual,
            dado por os.getcwd()
        full_path : bool
            Decide si los elementos de la lista son el contenido del directorio en cuestión,
            si se le agrega el nombre del directorio: ['capeta/subcarpeta1', 'capeta/subcarpeta2',
            'capeta/archivo.py']. Default: True
        natsort : bool
            Decide si ordenar o no los contenidos de la carpeta utilizando "orden natural", en vez
            de alfanumérico. Defualt = True
        filter_ext : None, str, iterable
            Filtra los contenidos de la lista por extensión. Si es None, no filta. Para quedarse
            sólo con los directorios, utilizar un string vacío ''. Para quedarse con varios tipos
            de archivos, utilizar una tupla de valores. Las extensiones deben ir con punto: '.jpg', 
            '.py', '.txt', etc.

    --------
    Métodos:
        update():
            Actualiza la lista si los contenidos de la carpeta cambiaron
        natural_sort():
            Ordena la lista según orden natural
        age_sort():
            Ordena la lista según última fecha de modificación
        print_orden(only_basename=True):
            Imprime los contenidos de la lista con los índices
        filtered_ext(ext):
            devuelve una NUEVA lista sólo con los elementos cuya extensión es ext
        files():
            devuelve una NUEVA lista sólo con los elementos que son archivos
        directories():
            devuelve una NUEVA lista sólo con los elementos que son directorios
        keep_files():
            modifica esta lista para quedarse sólo con los archivos
        keep_dirs():
            modifica esta lista para quedarse sólo con los directorios

    '''
    def __init__(self, carpeta=None, full_path=True, sort='natural', filter_ext=None):
        '''Si full_path=True, los elementos de la lista serán los contenidos de la carpeta
        apendeados con el nombre de la carpeta en sí. Si no, serán sólo los contenidos, en
        cuyo caso habrá algunas funconalidads no disponibles.'''

        self.carpeta = carpeta or os.getcwd()
        if not os.path.isdir(self.carpeta):
            raise NotADirectoryError(
                    'El nombre del directorio no es válido: '+ 
                    self.carpeta)

        self.sort_type = sort
        
        self._full_path = full_path # performs the sortng
        self.filter_ext = filter_ext #also performs the sorting

    @property
    def filter_ext(self):
        return self._filter_ext
    @filter_ext.setter
    def filter_ext(self, value):
        if isinstance(value, str):
            self._filter_ext_tup = (value,)
        elif hasattr(value, '__iter__'):
            self._filter_ext_tup = value
        elif value is None:
            pass 
        else:
            raise TypeError('filter_ext must be a string representing an extension, or an iterable with with many extensions')
        self._filter_ext = value
        self.update()

    @property
    def full_path(self):
        return self._full_path
    @full_path.setter
    def full_path(self, value):
        if not isinstance(value, bool):
            raise TypeError('full_path must be bool type')
        self._full_path = value
        self.update()
        
    @property
    def sort_type(self):
        return self._sort_type
    @sort_type.setter
    def sort_type(self, value):
        valid = ['natural', 'lexicographic', 'age', None]
        if value not in valid:
            raise ValueError("Sort must be one of ['natural', 'lexicographic', 'age' or None.]")
        self._sort_type = value

    def __sorter(self):
        sort = self._sort_type
        if sort=='natural':
            self.natural_sort()
        elif sort=='lexicographic':
            self.sort()
        elif sort=='age':
            self.age_sort()
        else:
            pass

    def update(self):
        if self.filter_ext is None:
            archivos = os.listdir(self.carpeta)
        else:
            archivos = (f for f in os.listdir(self.carpeta) if os.path.splitext(f)[-1] in self._filter_ext_tup)

        if self.full_path:
            super().__init__((os.path.join(self.carpeta, f) for f in archivos))
        else:
            super().__init__(archivos)
        
        self.__sorter()

    def natural_sort(self):
        convert = lambda text: int(text) if text.isdigit() else text.lower() 
        self.sort(key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])
    
    def age_sort(self):
        self.sort(key=os.path.getmtime)
        
    def print_orden(self, only_basename=True):
        if only_basename:
            for i, f in enumerate(self):
                print('{}: {}'.format(i, os.path.basename(f)))
        else:
            for i, f in enumerate(self):
                print('{}: {}'.format(i, f))

    def filtered_ext(self, extension):
        '''Crea una nueva lista de las cosas con la extensión correspondiente.'''
        return [elem for elem in self if elem.endswith(extension)]

    # def filter_ext(self, extension):
    #     '''Elimina de la lista todo lo que no tenga la extensión correspondiente'''
    #     super().__init__(self.filtered_ext(extension))

    def files(self):
        '''Devuelve nueva lista de sólo los elementos que son archivos.'''
        return [elem for elem in self if os.path.isfile(elem)]
        
    def keep_fies(self):
        '''Elimina de la lista todo lo que no sean archivos.'''
        super().__init__(self.files())

    def directories(self):
        '''Devuelve nueva lista de sólo los elementos que son carpetas.'''
        return [elem for elem in self if os.path.isdir(elem)]
    
    def keep_dirs(self):
        '''Elimina de la lista todo lo que no sean carpetas.'''
        super().__init__(self.directories())

class Testimado:

    def __init__(self, cant_total):
        self.cant_total = cant_total
        self.inicio = time.time()

    def restante(self, indice):
        return round((self.cant_total / (indice+1) - 1) * self.transcurrido())

    def transcurrido(self):
        return time.time() - self.inicio
    
    def horas_minutos(self, i):
         horas, rem = divmod(self.restante(i), 3600)
         minutos = rem//60
         return horas, minutos
         
    def horas_minutos_segundos(self, i):
         horas, rem = divmod(self.restante(i), 3600)
         minutos, segundos= divmod(rem, 60)
         return (horas, minutos, segundos)
     
    def time_str(self, i, include_times = 'HM'):
        '''Devuelve un string con el tiempo restante formateado según se indica
        en include_times.
        j: días
        H: horas
        M: minutos
        S: segundos'''
        format_string = ':'.join('%{}'.format(s) for s in include_times)
        return time.strftime(format_string, time.gmtime(self.restante(i)))
    
    def print_remaining(self, i, *a, **kw):
        print('ETA: {}'.format(self.time_str(i, *a, **kw)))

class Grid:
    '''Una clase para crear y llenar una grilla con imagenes. A menos que 
    especifique una cantidad de columnas o de filas, la grilla creada será 
    tan cuadrada como sea posible. Funciona con imágenes de un solo canal 
    (escala de grises) y de tres canales (RGB).
    
    -----------
    Parámetros:
        cant : int
            cantidad de imagenes que se introducirán en la grilla. Es necesario
            pasar este parámetro en la construcción de la clase para poder 
            planear el tamaño de la grilla.
        fill_with : int, float
            elemento con el que rellenar los lugares vacíos de la grilla. Afecta 
            cómo se verá la grilla al graficar. Default: np.nan
        trasponer : bool
            decide si trasponer la grilla calculada o no. Por defecto, si la 
            grilla no es cuadrada, tendrá más columnas que filas. Trasponer 
            invierte esto. Afecta también a grillas con cantidades persnoalizadas
            de filas y columnas. Default: False
        bordes : bool
            decide si poner bordes entre las imagenes y márgenes en torno a la 
            imagen. Default: True
        cant_col : int
            fija cantidad de columnas a utilizar, sobreescribiendo la cantidad
            calculada para optimizar cuadradez de la grilla. Si cant_col=None, 
            utilizará la calulada. Default: None
        cant_row : int
            fija cantidad de filas a utilizar, sobreescribiendo la cantidad
            calculada para optimizar cuadradez de la grilla. Si cant_row=None, 
            utilizará la calulada. Default: None
            
    --------
    Métodos:
        insert_image(im):
            Inserta una imagen a la grilla
        show(**kwargs):
            Muestra la grilla en su estado actual utilizando plt.imshow(). 
            Pasa los argumentos extra a imshow.
    '''
       
    def __init__(self, cant, fill_with=np.nan, trasponer=False, 
                 bordes=True, cant_col=None, cant_row=None ):

        self.cant = cant #cantidad de imagenes
        self.trasponer = trasponer #por default, la grilla es más ancha que alta
        self.bordes = bordes #si debe o no haber un margen entre figus.
        
        if cant_col is not None and cant_row is not None:
            raise ValueError('Sólo se puede especificar cantidad de filas o cantidad de columnas. La otra será deducida de la cantidad de elementos')
            
        self.shape = self._cant_to_mat(cant_col, cant_row) #tamaño de la matriz de imagenes

        self.grid = None #la grilla a llenar con imagenes
        #self.im_shape = None #tamaño de la imagen
        self.ind = 0 #por qué imagen voy?
        self.fill_with = fill_with #con qué lleno la grilla vacía

    @property
    def im_shape(self):
        return self._im_shape_real
    @im_shape.setter
    def im_shape(self, value):
        self._im_shape_real = value
        self._imRGB = len(value)==3 #if image is RGB
        
        if self.bordes:
            self._im_shape_bordes = (value[0] + 1, value[1] + 1)
        else:
            self._im_shape_bordes = self.im_shape

    def _cant_to_mat(self, cant_col, cant_row):
        '''Dimensiones de la cuadrícula más pequeña y más cuadrada
        posible que puede albergar [self.cant] cosas.'''
        
        if cant_col is not None:
            col = int(cant_col)
            row = int(np.ceil(self.cant/col))
        elif cant_row is not None:
            row = int(cant_row)
            col = int(np.ceil(self.cant/row))
        else:
            col = int(np.ceil(np.sqrt(self.cant)))
            row = int(round(np.sqrt(self.cant)))
            
        if self.trasponer:
            return col, row
        else:
            return row, col
        
    def _filcol(self):
        '''Pasa de índice lineal a matricial.'''
        fil = self.ind // self.shape[1]
        col = self.ind % self.shape[1]
        return int(fil), int(col)

    def _create_grid(self):
        shape = (self._im_shape_bordes[0] * self.shape[0], 
                 self._im_shape_bordes[1] * self.shape[1])
        if self.bordes:
            shape = shape[0] + 1, shape[1] + 1
        if self._imRGB:
            shape = *shape, 3
            
        self.grid = np.full(shape, self.fill_with)

    def insert_image(self, im):
        '''Agrego una imagen a la grilla.'''
        #inicializo la grilla
        if self.grid is None:
            self.im_shape = im.shape
            self._create_grid()

        #la lleno
        col, row = self._filcol()
        #sumo el booleano de bordes para que cierre bien la cuenta
        if self._imRGB:
            self.grid[col * self._im_shape_bordes[0] + int(self.bordes) : 
                        (col + 1) * self._im_shape_bordes[0],
                    row * self._im_shape_bordes[1] + int(self.bordes) : 
                        (row + 1) * self._im_shape_bordes[1], :]= im
        else:        
            self.grid[col * self._im_shape_bordes[0] + int(self.bordes) : 
                        (col + 1) * self._im_shape_bordes[0],
                    row * self._im_shape_bordes[1] + int(self.bordes) : 
                        (row + 1) * self._im_shape_bordes[1]]= im
        
        #avanzo el contador apra la siguiente imagen
        self.ind += 1
        
    def show(self, **kw):
        plt.imshow(self.grid, cmap=kw.pop('cmap', 'viridis'), **kw)

#Grid Testing
# cant = 33
# shape = (21,25)
# g = Grid(cant, trasponer=False, bordes=True, fill=100)
# for i in range(cant):
#     g.insert_image(np.ones(shape)*i)

# plt.matshow(g.grid)
# plt.grid()

# #%%

# cant = 17
# shape = (11,9)
# g = Grid(cant, trasponer=False, bordes=True, fill=np.nan)
# colores = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,0), 
#            (1,1,1), (.5,.5,.5), (1,.5,0), (1,0,.5), (.5,1,0), (.5,0,1),
#            (0,.5,1), (0,1,.5), (.5,0,0), (0,.5,0), (0,0,.5)]
# imagenes = []
# for c in colores:
#     liso = np.ones((*shape,3))
#     for i in range(3):
#         liso[:,:,i] *= c[i]
#     imagenes.append(liso)

# for i in range(cant):
#     g.insert_image(imagenes[i])

# plt.imshow(g.grid)
# plt.grid()

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        import warnings
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `load_img` requires PIL.')
    img = pil_image.open(path)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img

