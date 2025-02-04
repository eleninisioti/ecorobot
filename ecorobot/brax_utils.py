from xml.etree import ElementTree
import jax.numpy as np
import math
from typing import Dict, Optional, Tuple, Union

def _transform_do(
    parent_pos: np.ndarray, parent_quat: np.ndarray, pos: np.ndarray,
    quat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  pos = parent_pos + math.rotate_np(pos, parent_quat)
  rot = math.quat_mul_np(parent_quat, quat)
  return pos, rot
def _offset(
    elem: ElementTree.Element, parent_pos: np.ndarray, parent_quat: np.ndarray):
  """Offsets an element."""
  pos = elem.attrib.get('pos', '0 0 0')
  quat = elem.attrib.get('quat', '1 0 0 0')
  pos = np.fromstring(pos, sep=' ')
  quat = np.fromstring(quat, sep=' ')
  fromto = elem.attrib.get('fromto', None)
  if fromto:
    # fromto attributes are not compatible with pos/quat attributes
    from_pos = np.fromstring(' '.join(fromto.split(' ')[0:3]), sep=' ')
    to_pos = np.fromstring(' '.join(fromto.split(' ')[3:6]), sep=' ')
    from_pos, _ = _transform_do(parent_pos, parent_quat, from_pos, quat)
    to_pos, _ = _transform_do(parent_pos, parent_quat, to_pos, quat)
    fromto = ' '.join('%f' % i for i in np.concatenate([from_pos, to_pos]))
    elem.attrib['fromto'] = fromto
    return
  pos, quat = _transform_do(parent_pos, parent_quat, pos, quat)
  pos = ' '.join('%f' % i for i in pos)
  quat = ' '.join('%f' % i for i in quat)
  elem.attrib['pos'] = pos
  elem.attrib['quat'] = quat

def fuse_bodies(elem: ElementTree.Element):
  """Fuses together parent child bodies that have no joint."""

  for child in list(elem):  # we will modify elem children, so make a copy
    _fuse_bodies(child)
    # this only applies to bodies with no joints
    if child.tag != 'body':
      continue
    if child.find('joint') is not None or child.find('freejoint') is not None:
      continue
    cpos = child.attrib.get('pos', '0 0 0')
    cpos = np.fromstring(cpos, sep=' ')
    cquat = child.attrib.get('quat', '1 0 0 0')
    cquat = np.fromstring(cquat, sep=' ')
    for grandchild in child:
      # TODO: might need to offset more than just (body, geom)
      if grandchild.tag in ('body', 'geom') and (cpos != 0).any():
        _offset(grandchild, cpos, cquat)
      elem.append(grandchild)
    elem.remove(child)