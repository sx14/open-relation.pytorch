import os
import shutil
import xml.dom.minidom


def output_pascal_format(mid_data, output_path):
    # mid_data:
    # filename
    # width
    # height
    # depth
    # objects
    #   -- xmin
    #   -- ymin
    #   -- xmax
    #   -- ymax
    #   -- name
    #   -- pose
    #   -- truncated
    #   -- difficult
    additional_data = dict()
    additional_data['folder'] = 'VOC2007'
    additional_data['s_database'] = 'The VOC2007 Database'
    additional_data['s_annotation'] = 'PASCAL VOC2007'
    additional_data['s_image'] = 'flickr'
    additional_data['s_flickrid'] = '123456789'
    additional_data['o_flickrid'] = 'Tom'
    additional_data['o_name'] = 'Tom'
    additional_data['segmented'] = '0'
    des_xml_dom = xml.dom.minidom.Document()
    # annotation
    des_root_node = des_xml_dom.createElement('annotation')
    # folder
    des_folder_node = des_xml_dom.createElement('folder')
    des_folder = des_xml_dom.createTextNode(additional_data['folder'])
    des_folder_node.appendChild(des_folder)
    des_root_node.appendChild(des_folder_node)
    # filename
    des_filename_node = des_xml_dom.createElement('filename')
    des_filename = des_xml_dom.createTextNode(mid_data['filename'])
    des_filename_node.appendChild(des_filename)
    des_root_node.appendChild(des_filename_node)
    # source
    des_dataset_name = des_xml_dom.createTextNode(additional_data['s_database'])
    des_dataset_node = des_xml_dom.createElement('database')
    des_dataset_node.appendChild(des_dataset_name)
    des_annotation = des_xml_dom.createTextNode(additional_data['s_annotation'])
    des_annotation_node = des_xml_dom.createElement('annotation')
    des_annotation_node.appendChild(des_annotation)
    des_image = des_xml_dom.createTextNode(additional_data['s_image'])
    des_image_node = des_xml_dom.createElement('image')
    des_image_node.appendChild(des_image)
    des_flickrid = des_xml_dom.createTextNode(additional_data['s_flickrid'])
    des_flickrid_node = des_xml_dom.createElement('flickrid')
    des_flickrid_node.appendChild(des_flickrid)
    des_source_node = des_xml_dom.createElement('source')
    des_source_node.appendChild(des_dataset_node)
    des_source_node.appendChild(des_annotation_node)
    des_source_node.appendChild(des_image_node)
    des_source_node.appendChild(des_flickrid_node)
    des_root_node.appendChild(des_source_node)
    # owner
    des_owner_flickrid = des_xml_dom.createTextNode(additional_data['o_flickrid'])
    des_owner_flickrid_node = des_xml_dom.createElement('flickrid')
    des_owner_flickrid_node.appendChild(des_owner_flickrid)
    des_owner_name = des_xml_dom.createTextNode(additional_data['o_name'])
    des_owner_name_node = des_xml_dom.createElement('name')
    des_owner_name_node.appendChild(des_owner_name)
    des_owner_node = des_xml_dom.createElement('owner')
    des_owner_node.appendChild(des_owner_flickrid_node)
    des_owner_node.appendChild(des_owner_name_node)
    des_root_node.appendChild(des_owner_node)
    # size
    des_size_node = des_xml_dom.createElement('size')
    des_width_node = des_xml_dom.createElement('width')
    des_height_node = des_xml_dom.createElement('height')
    des_depth_node = des_xml_dom.createElement('depth')
    des_width = des_xml_dom.createTextNode(str(mid_data['width']))
    des_height = des_xml_dom.createTextNode(str(mid_data['height']))
    des_depth = des_xml_dom.createTextNode(str(mid_data['depth']))
    des_width_node.appendChild(des_width)
    des_height_node.appendChild(des_height)
    des_depth_node.appendChild(des_depth)
    des_size_node.appendChild(des_width_node)
    des_size_node.appendChild(des_height_node)
    des_size_node.appendChild(des_depth_node)
    des_root_node.appendChild(des_size_node)
    # segmented
    des_segmented = des_xml_dom.createTextNode(additional_data['segmented'])
    des_segmented_node = des_xml_dom.createElement('segmented')
    des_segmented_node.appendChild(des_segmented)
    des_root_node.appendChild(des_segmented_node)
    # object
    org_objects = mid_data['objects']
    for j in range(0, len(org_objects)):
        org_object = org_objects[j]
        des_object_node = des_xml_dom.createElement('object')
        x_min = int(org_object['xmin'])
        y_min = int(org_object['ymin'])
        x_max = int(org_object['xmax'])
        y_max = int(org_object['ymax'])
        if x_min <= 0:
            org_object['xmin'] = '1'
        if y_min <= 0:
            org_object['ymin'] = '1'

        # name
        des_object_name = des_xml_dom.createTextNode(org_object['name'])
        des_object_name_node = des_xml_dom.createElement('name')
        des_object_name_node.appendChild(des_object_name)
        des_object_node.appendChild(des_object_name_node)
        # pose
        des_pose = des_xml_dom.createTextNode(org_object['pose'])
        des_pose_node = des_xml_dom.createElement('pose')
        des_pose_node.appendChild(des_pose)
        des_object_node.appendChild(des_pose_node)
        # truncated
        des_truncated = des_xml_dom.createTextNode(str(org_object['truncated']))
        des_truncated_node = des_xml_dom.createElement('truncated')
        des_truncated_node.appendChild(des_truncated)
        des_object_node.appendChild(des_truncated_node)
        # difficult
        des_object_difficult = des_xml_dom.createTextNode(str(org_object['difficult']))
        des_object_difficult_node = des_xml_dom.createElement('difficult')
        des_object_difficult_node.appendChild(des_object_difficult)
        des_object_node.appendChild(des_object_difficult_node)
        # bndbox
        des_xmin_node = des_xml_dom.createElement('xmin')
        des_xmin = des_xml_dom.createTextNode(str(org_object['xmin']))
        des_xmin_node.appendChild(des_xmin)
        des_ymin_node = des_xml_dom.createElement('ymin')
        des_ymin = des_xml_dom.createTextNode(str(org_object['ymin']))
        des_ymin_node.appendChild(des_ymin)
        des_xmax_node = des_xml_dom.createElement('xmax')
        des_xmax = des_xml_dom.createTextNode(str(org_object['xmax']))
        des_xmax_node.appendChild(des_xmax)
        des_ymax_node = des_xml_dom.createElement('ymax')
        des_ymax = des_xml_dom.createTextNode(str(org_object['ymax']))
        des_ymax_node.appendChild(des_ymax)
        des_object_box_node = des_xml_dom.createElement('bndbox')
        des_object_box_node.appendChild(des_xmin_node)
        des_object_box_node.appendChild(des_ymin_node)
        des_object_box_node.appendChild(des_xmax_node)
        des_object_box_node.appendChild(des_ymax_node)
        des_object_node.appendChild(des_object_box_node)
        des_root_node.appendChild(des_object_node)
    with open(output_path, 'w') as des_file:
        des_root_node.writexml(des_file, addindent='\t', newl='\n')