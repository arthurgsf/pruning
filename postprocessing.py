import SimpleITK as sitk
import numpy as np

def biggest_3D_object(volume):
    volume = volume.astype(np.int8)
    image_sitk = sitk.GetImageFromArray(volume)
    image_sitk.SetOrigin((0, 0, 0))

    connected_filter = sitk.ConnectedComponentImageFilter()
    connected_filter.FullyConnectedOn()
    new_image = connected_filter.Execute(image_sitk)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(new_image)
    maior_label=None
    qtd_maior_label=0
    for label in stats.GetLabels():
        if(stats.GetNumberOfPixels(label)>qtd_maior_label):
            maior_label=label
            qtd_maior_label=stats.GetNumberOfPixels(label)

    new_image_array=sitk.GetArrayFromImage(new_image)
    new_image_array[new_image_array!=maior_label]=0
    new_image_array[new_image_array==maior_label]=255
    new_image = sitk.GetImageFromArray(new_image_array)
    new_image.CopyInformation(image_sitk)
    return sitk.GetArrayFromImage(new_image).astype(np.float32)