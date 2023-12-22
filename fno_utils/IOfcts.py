import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import torch
import shutil
import re
import os


# =====================
# some helper functions
# =====================
def vtkFieldReader(vtk_name, fieldName='tomo_Volume'):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(vtk_name)
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    siz = list(dim)
    siz = [i - 1 for i in siz]
    mesh = vtk_to_numpy(data.GetCellData().GetArray(fieldName))
    return mesh.reshape(siz, order='F')

def read_macroStressStrain(fname):
    with open(fname) as f:
        lines = f.readlines()
    data = list()
    for line in lines[6:]:
        data.append( [float(num) for num in line.split()] )
    return np.array(data)


def registerFileName(lst_stress=None, lst_strain=None, lst_damage=None, fprefix=None, loadstep=None, zeroVTK=False):
    if zeroVTK is False:
        if lst_stress is not None:
            for key in lst_stress:
                lst_stress[key].append(fprefix + '_' + key + '_' + str(loadstep) + '.vtk')

        if lst_strain is not None:
            for key in lst_strain:
                lst_strain[key].append(fprefix + '_' + key + '_' + str(loadstep) + '.vtk')

        if lst_damage is not None:
            for key in lst_damage:
                lst_damage[key].append(fprefix + '_' + key + '_' + str(loadstep) + '.vtk')
    else:

        zeroVTKlocation = 'zerovtk.vtk'

        if lst_stress is not None:
            for key in lst_stress:
                lst_stress[key].append(zeroVTKlocation)

        if lst_strain is not None:
            for key in lst_strain:
                lst_strain[key].append(zeroVTKlocation)

        if lst_damage is not None:
            for key in lst_damage:
                lst_damage[key].append(zeroVTKlocation)


def init_dict_StressStrainDamage():
    dict_stress = {'sig1': list(),
                   'sig2': list(),
                   'sig4': list()}
    dict_strain = {'def1': list(),
                   'def2': list(),
                   'def4': list()}
    dict_damage = {'M1_varInt1': list(),
                   'M2_varInt1': list()}

    return dict_stress, dict_strain, dict_damage

def vtk_field_name(key):
    if key == 'sig1':
        return 'Sig_1'
    elif key == 'sig2':
        return 'Sig_2'
    elif key== 'sig4':
        return 'Sig_4'
    elif key == 'def1':
        return 'Def_1'
    elif key == 'def2':
        return 'Def_2'
    elif key == 'def4':
        return 'Def_4'
    elif key== 'M1_varInt1':
        return 'M1_varInt1'
    elif key == 'M2_varInt1':
        return 'M2_varInt1'
    else:
        assert "key unknown, sorry"



import matplotlib.pyplot as plt
def plot_macro_vtksteps(data_macro, idx, Ieps=7, Isig=1):
    plt.plot(data_macro[:,Ieps], data_macro[:,Isig])
    plt.plot(data_macro[idx[:]-1,Ieps], data_macro[idx[:]-1,Isig], 'o')

    plt.show()


# ==================
# read vtk mesh data
# ==================
# class meshvtkDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.transform = transform
#         self.root_dir = root_dir
#         self.files = datasets.utils.list_files(root_dir, ".vtk")
#         self.files = sorted(self.files, key=lambda x: float(re.search('iUC(\d+)', x).group(1)))
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         img_name = self.root_dir + '/' + self.files[idx]
#         reader = vtk.vtkStructuredPointsReader()
#         reader.SetFileName(img_name)
#         reader.Update()
#         data = reader.GetOutput()
#         dim = data.GetDimensions()
#         siz = list(dim)
#         siz = [i - 1 for i in siz]
#         mesh = vtk_to_numpy(data.GetCellData().GetArray('tomo_Volume'))
#         mesh = (mesh.reshape(siz, order='F') - 1) * 255
#
#         if self.transform:
#             mesh = self.transform(mesh)
#
#         return mesh, mesh
class meshvtkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.files = datasets.utils.list_files(root_dir, ".vtk")
        self.files = sorted(self.files, key=lambda x: float(re.search('iUC(\d+)', x).group(1)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir + '/' + self.files[idx]
        mesh = vtkFieldReader(img_name, 'tomo_Volume')
        mesh = (mesh - 1) * 255

        # # multi-class
        # if np.unique(mesh).min()<=0:
        #     raise ValueError('all labels in the VTK mesh must be larger than 0')
        # label = np.array(mesh)
        # for idx in np.unique(mesh):
        #     label = np.append(label, (mesh==idx), axis=-1)
        # mesh = label[:, :, 1:]

        if self.transform:
            mesh = self.transform(mesh)

        return mesh, mesh


# ====================================================
# import data - structure for {stress/strain -> crack}
# ====================================================
class elastic2crackDataset(Dataset):
    def __init__(self, lst_input_dir, lst_output_dir, lst_in_field_name, lst_out_field_name, transform=None):
        self.transform = transform
        self.lst_input_dir = lst_input_dir
        self.lst_output_dir = lst_output_dir
        self.lst_in_field_name = lst_in_field_name
        self.lst_out_field_name = lst_out_field_name

        # filenames for inputs
        self.input_files = list()
        for input_dir in lst_input_dir:
            tmp = datasets.utils.list_files(input_dir, ".vtk")
            tmp = sorted(tmp, key=lambda x: float(re.search('iUC(\d+)', x).group(1)))
            self.input_files.append( tmp )

        # filenames for outputs
        self.output_files = list()
        for output_dir in lst_output_dir:
            tmp = datasets.utils.list_files(output_dir, ".vtk")
            tmp = sorted(tmp, key=lambda x: float(re.search('iUC(\d+)', x).group(1)))
            self.output_files.append( tmp )

    def __len__(self):
        return len(self.input_files[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #input data
        inputs = list()
        for i, input_dir in enumerate(self.lst_input_dir):
            fname = input_dir + '/' + self.input_files[i][idx]
            input = vtkFieldReader(fname, self.lst_in_field_name[i])
            inputs.append( input )
        inputs = np.dstack( inputs )
        
        # output data
        outputs = list()
        for i, output_dir in enumerate(self.lst_output_dir):
            fname = output_dir + '/' + self.output_files[i][idx]
            output = vtkFieldReader(fname, self.lst_out_field_name[i])
            outputs.append( output )
        outputs = np.dstack( outputs )

        # #preprocess input data
        # # ->normalisation
        # avVALs = np.array([ np.average(inputs[:,:,i].flatten()) for i in range(6) ])
        # avDEFs, avSIGs = avVALs[:3], avVALs[3:]
        # # avDEFmax = avDEFs[np.argmax(np.abs(avDEFs))] #max absolute value
        # # avSIGmax = avSIGs[np.argmax(np.abs(avSIGs))] #max absolute value
        # avDEFmax, avSIGmax = np.max(np.abs(avDEFs)), np.max(np.abs(avSIGs))
        # tmp = [ inputs[:,:,i] / avDEFmax for i in range(3)]
        # inputs[:, :, :3] = np.moveaxis(tmp, 0, -1)
        # tmp = [ inputs[:,:,i] / avSIGmax for i in np.array([3,4,5])]
        # inputs[:, :, 3:] = np.moveaxis(tmp, 0, -1)

        #preprocess input data
        # --> calculate strain energy
        energy = inputs[:,:,0] * inputs[:,:,3] + inputs[:,:,1] * inputs[:,:,4] + 2* inputs[:,:,2] * inputs[:,:,5]
        inputs = energy / energy.max()
        # energy = energy / energy.max()
        # inputs = np.dstack( (energy, inputs[:,:,6:]) )

        #preprocess output data
        # ->non, for now
        
        if self.transform:
            inputs = self.transform(inputs)
            outputs = self.transform(outputs)

        return inputs, outputs



# ====================================================================
# read vtk simulation results, regroup stress/strain tensor components
# ====================================================================
class multichannelDataset(Dataset):
    def __init__(self, mesh_dir, output_root, lst_comp, in_field_name='tomo_Volume', out_field_prefix='Sig_', transform=None):
        self.transform = transform
        self.mesh_dir = mesh_dir
        self.output_root = output_root
        self.lst_comp = lst_comp
        self.in_field_name = in_field_name
        self.out_field_prefix = out_field_prefix
        self.mesh = datasets.utils.list_files(mesh_dir, ".vtk")
        self.mesh = sorted(self.mesh, key=lambda x: float(re.search('iUC(\d+)', x).group(1)))
        self.output = []
        for i, comp in enumerate(self.lst_comp):
            self.output.append( datasets.utils.list_files(output_root+'\sig'+str(comp), ".vtk") )
            self.output[i] = sorted(self.output[i], key=lambda x: float(re.search('iUC(\d+)', x).group(1)))

    def __len__(self):
        return len(self.mesh)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.mesh_dir + '/' + self.mesh[idx]
        mesh = vtkFieldReader(img_name, self.in_field_name)
        mesh = (mesh - 1) * 255

        out = list()
        for i, comp in enumerate(self.lst_comp):
            img_name = self.output_root + '/sig' + str(comp) + '/' + self.output[i][idx]
            out.append( vtkFieldReader(img_name, self.out_field_prefix + str(comp)) )
        out = np.dstack( out )

        if self.transform:
            mesh = self.transform(mesh)
            out = self.transform(out)

        return mesh, out


# ===========================
# read vtk simulation results
# ===========================
class simulationDataset(Dataset):
    def __init__(self, mesh_dir, output_dir, in_field_name='tomo_Volume', out_field_name='M2_varInt1', transform=None):
        self.transform = transform
        self.mesh_dir = mesh_dir
        self.output_dir = output_dir
        self.in_field_name = in_field_name
        self.out_field_name = out_field_name
        self.mesh = datasets.utils.list_files(mesh_dir, ".vtk")
        self.mesh = sorted(self.mesh, key=lambda x: float(re.search('iUC(\d+)', x).group(1)))
        self.output = datasets.utils.list_files(output_dir, ".vtk")
        self.output = sorted(self.output, key=lambda x: float(re.search('iUC(\d+)', x).group(1)))

    def __len__(self):
        return len(self.mesh)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.mesh_dir + '/' + self.mesh[idx]
        mesh = vtkFieldReader(img_name, self.in_field_name)
        mesh = (mesh - 1) * 255

        img_name = self.output_dir + '/' + self.output[idx]
        output = vtkFieldReader(img_name, self.out_field_name)

        if self.transform:
            mesh = self.transform(mesh)
            output = self.transform(output)

        return mesh, output


# ===========================================
# read vtk simulation results for autoencoder
# ===========================================
class simulationDataset_AE(Dataset):
    def __init__(self, output_dir, out_field_name='M2_varInt1', transform=None):
        self.transform = transform
        self.output_dir = output_dir
        self.out_field_name = out_field_name
        self.output = datasets.utils.list_files(output_dir, ".vtk")
        self.output = sorted(self.output, key=lambda x: float(re.search('iUC(\d+)', x).group(1)))

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.output_dir + '/' + self.output[idx]
        output = vtkFieldReader(img_name, self.out_field_name)

        if self.transform:
            output = self.transform(output)

        return output, output






# ====================================================
# import data - structure for full-history prediction
# ====================================================
class historyDataset(Dataset):
    def __init__(self, dir_mesh, lst_dir0=None, LOADprefix='tensX', Isig=1, Ieps=7, transform=None):
        self.transform = transform

        # create empty lists for input / output
        in_stress = list()
        in_strain = list()
        in_damage = list()
        out_stress = list()
        out_strain = list()
        out_damage = list()
        in_MESH = list()
        in_LOAD = list()
        files_macro = list()
        bad_results = list()

        # loop over directories - different vf, different UC, vp
        for dir0 in lst_dir0:  #different vf

            for dir1 in os.listdir(dir0):  #different UC, vp

                # only unit cells with void are used
                if "vpMIN" not in dir1:
                    continue

                # initial geometry - mesh
                iuc = re.findall(r"\d+", dir1)[0]
                img_name = dir_mesh + '/' + dir1 + '.vtk'
                # mesh = vtkFieldReader(img_name, fieldName='tomo_Volume')

                # indices of vtk-steps
                tmp = datasets.utils.list_files(dir0 + '/' + dir1, ".vtk")
                idx = list()
                for tmpi in tmp:
                    idx.append([int(num) for num in re.findall(r"(\d+).vtk", tmpi)])
                idx = np.unique(idx)

                # stress-strain curves
                file_macro = datasets.utils.list_files(dir0 + '/' + dir1, ".std")[0]
                files_macro.append( dir0 + '/' + dir1 + '/' + file_macro )
                data_macro = read_macroStressStrain(dir0 + '/' + dir1 + '/' + file_macro)
                idx_peak = np.argmax(data_macro[:, Isig])
                idx_max = idx.max()
                # plot_macro_vtksteps(data_macro, idx, Ieps=7, Isig=1)

                #sanity check
                if idx_max > data_macro.shape[0]:
                    bad_results.append( dir0 + '/' + dir1 )
                    continue

                # input: mesh, load, strain_elas, stress_elas, damage_elas
                in_MESH.append( dir_mesh + '/' + dir1 + '.vtk' )
                in_LOAD.append( data_macro[idx-1, Ieps] )
                tmp_stress, tmp_strain, tmp_damage = init_dict_StressStrainDamage()
                registerFileName(tmp_stress, tmp_strain, tmp_damage, dir0 + '/' + dir1 + '/' + LOADprefix, 10)
                in_stress.append( tmp_stress )
                in_strain.append( tmp_strain )
                in_damage.append( tmp_damage )

                # output: stress, strain, damage
                tmp_stress, tmp_strain, tmp_damage = init_dict_StressStrainDamage()
                for loadstep in idx[1:]:
                    registerFileName(tmp_stress, tmp_strain, tmp_damage, dir0 + '/' + dir1 + '/' + LOADprefix, loadstep)
                out_stress.append( tmp_stress )
                out_strain.append( tmp_strain )
                out_damage.append( tmp_damage )

        self.in_MESH    = in_MESH
        self.in_LOAD    = in_LOAD
        self.in_stress  = in_stress
        self.in_strain  = in_strain
        self.in_damage  = in_damage
        self.out_stress = out_stress
        self.out_strain = out_strain
        self.out_damage = out_damage
        self.files_macro = files_macro

    def __len__(self):
        return len(self.in_MESH)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = [list(), list(), np.array([]).reshape(501,501,0), np.zeros((501,501,1))]
        output = [list(), list()]

        # input[0] - mesh
        input[0] = vtkFieldReader(self.in_MESH[idx], 'tomo_Volume')

        # input[1] - load
        input[1] = np.array(self.in_LOAD[idx])

        # input[2] - stress
        for key in self.in_stress[idx]:
            input[2] = np.concatenate((input[2], vtkFieldReader(self.in_stress[idx][key][0], fieldName=vtk_field_name(key))), axis=2)

        # input[3] - damage
        for key in self.in_damage[idx]:
            input[3] += vtkFieldReader(self.in_damage[idx][key][0], fieldName=vtk_field_name(key))

        # output[0] - stress[step]
        nsteps = len(self.out_stress[idx]['sig1'])
        for step in range(nsteps):
            tmp = np.array([]).reshape(501, 501, 0)
            for key in self.out_stress[idx]:
                tmp = np.concatenate((tmp, vtkFieldReader(self.out_stress[idx][key][step], fieldName=vtk_field_name(key))), axis=2)
            output[0].append( tmp )

        # output[1] - damage[step]
        for step in range(nsteps):
            tmp = np.zeros((501,501,1))
            for key in self.out_damage[idx]:
                tmp += vtkFieldReader(self.out_damage[idx][key][step], fieldName=vtk_field_name(key))
            output[1].append( tmp )


        # transform - data augmentation
        if self.transform:
            transform_det = self.transform.to_deterministic()
            input[0] = torch.from_numpy(np.moveaxis(transform_det(image=input[0]),-1,0))
            input[2] = torch.from_numpy(np.moveaxis(transform_det(image=input[2]),-1,0))
            input[3] = torch.from_numpy(np.moveaxis(transform_det(image=input[3]),-1,0))
            for step in range(nsteps):
                output[0][step] = torch.from_numpy(np.moveaxis(transform_det(image=output[0][step]),-1,0))
                output[1][step] = torch.from_numpy(np.moveaxis(transform_det(image=output[1][step]),-1,0))

        # macro stress-strain data
        data_macro = read_macroStressStrain(self.files_macro[idx])

        return input, output, data_macro



# ==================================================================================================
# import data - structure for full-history prediction, for new data with a smaller dimension (L0.05)
# ==================================================================================================
class sequenceDataset(Dataset):
    def __init__(self, dir_mesh, lst_dir0=None, LOADprefix='Load0.0', Isig=1, Ieps=7, transform=None, outputFields='stressANDdamage'):
        self.transform = transform

        # create empty lists for input / output
        out_stress = list()
        out_strain = list()
        out_damage = list()
        in_MESH = list()
        in_LOAD = list()
        files_macro = list()
        bad_results = list()
        load_max_overall = 0 #maximum load
        num_steps_max = 0 #maximum number of vtk-steps
        num_steps = list()

        # loop over directories - different vf, different UC, vp
        keyword = 'L0.05_vf'
        for dir0 in lst_dir0:  #different vf
            VF = re.search(f"{keyword}.*?(\d+\.\d+)", dir0).group(1)

            for dir1 in os.listdir(dir0):  #different UC, vp

                # initial geometry - mesh
                iuc = re.findall(r"\d+", dir1)[0]
                img_name = dir_mesh + '/' + keyword + VF + '/' + 'h0.0002' + '/' + dir1 + '.vtk'

                # indices of vtk-steps
                tmp = datasets.utils.list_files(dir0 + '/' + dir1, ".vtk")
                idx = list()
                for tmpi in tmp:
                    idx.append([int(num) for num in re.findall(r"(\d+).vtk", tmpi)])
                idx = np.unique(idx)

                # number of steps
                num_steps.append(len(idx))

                # stress-strain curves
                file_macro = datasets.utils.list_files(dir0 + '/' + dir1, ".std")[0]
                files_macro.append( dir0 + '/' + dir1 + '/' + file_macro )
                data_macro = read_macroStressStrain(dir0 + '/' + dir1 + '/' + file_macro)
                idx_peak = np.argmax(data_macro[:, Isig])
                idx_max = idx.max()
                # plot_macro_vtksteps(data_macro, idx, Ieps=7, Isig=1)
                if data_macro[idx_max-1,Ieps] > load_max_overall:
                    load_max_overall = data_macro[idx_max-1,Ieps]

                #sanity check
                if idx_max > data_macro.shape[0]:
                    bad_results.append( dir0 + '/' + dir1 )
                    continue

                # input: mesh, load, strain_elas, stress_elas, damage_elas
                in_MESH.append( img_name )
                in_LOAD.append( data_macro[idx-1, Ieps] )

                # output: stress, strain, damage
                tmp_stress, tmp_strain, tmp_damage = init_dict_StressStrainDamage()
                for loadstep in idx:
                    registerFileName(tmp_stress, tmp_strain, tmp_damage, dir0 + '/' + dir1 + '/' + LOADprefix, loadstep)
                out_stress.append( tmp_stress )
                out_strain.append( tmp_strain )
                out_damage.append( tmp_damage )

        self.in_MESH    = in_MESH
        self.in_LOAD    = in_LOAD
        self.out_stress = out_stress
        self.out_strain = out_strain
        self.out_damage = out_damage
        self.files_macro = files_macro
        self.max_num_steps = max(num_steps)
        self.num_steps = num_steps
        self.outputFields = outputFields

    def __len__(self):
        return len(self.in_MESH)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # mesh
        mesh = vtkFieldReader(self.in_MESH[idx], 'tomo_Volume')

        # load
        load = torch.from_numpy(np.array(self.in_LOAD[idx]))

        # output - stress or/and damage
        height, width, _ = mesh.shape

        if self.outputFields=='stressANDdamage':
            output = np.zeros((self.num_steps[idx], 4, height, width))
            i = 0
            for key in self.out_stress[idx]:
                for step in range(self.num_steps[idx]):
                    output[step,i,:,:] = vtkFieldReader(self.out_stress[idx][key][step], fieldName=vtk_field_name(key))[:,:,0]
                i += 1
            for step in range(self.num_steps[idx]):
                tmp = np.zeros((251, 251, 1))
                for key in self.out_damage[idx]:
                    tmp += vtkFieldReader(self.out_damage[idx][key][step], fieldName=vtk_field_name(key))
                output[step,3,:,:] = tmp[:,:,0]

        elif self.outputFields=='damage':
            output = np.zeros((self.num_steps[idx], 1, height, width))
            for step in range(self.num_steps[idx]):
                tmp = np.zeros((251,251,1))
                for key in self.out_damage[idx]:
                    tmp += vtkFieldReader(self.out_damage[idx][key][step], fieldName=vtk_field_name(key))
                output[step, 0, :, :] = tmp[:, :, 0]

        elif self.outputFields=='stress':
            output = np.zeros((self.num_steps[idx], 3, height, width))
            i = 0
            for key in self.out_stress[idx]:
                for step in range(self.num_steps[idx]):
                    output[step,i,:,:] = vtkFieldReader(self.out_stress[idx][key][step], fieldName=vtk_field_name(key))[:,:,0]
                i += 1

        # transform - data augmentation
        if self.transform:
            transform_det = self.transform.to_deterministic()
            mesh = torch.from_numpy(np.moveaxis(transform_det(image=mesh),-1,0))
            for i in range(output.shape[1]):
                for step in range(self.num_steps[idx]):
                    output[step,i,:,:] = transform_det(image=output[step,i,:,:])
            output = torch.from_numpy(output)

        # macro stress-strain data
        data_macro = read_macroStressStrain(self.files_macro[idx])

        return mesh, load, output, data_macro


from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
def pad_collate(batch_data):
    (mesh, load, output, data_macro) = zip(*batch_data)

    # stack the meshes
    mesh = torch.stack(mesh) / 3

    # pack padded output fields
    output_lens = [x.shape[0] for x in output]
    output = pad_sequence(output, batch_first=True, padding_value=0)
    # output = pack_padded_sequence(output, output_lens)

    return mesh, load, output, output_lens, data_macro








# ==================================================================================================
# import data - structure for full-history prediction, for new data with a smaller dimension (L0.05)
# ==================================================================================================
class videoDataset(Dataset):
    def __init__(self, dir_mesh, lst_dir0=None, LOADprefix='Load0.0', transform=None, outputFields='damage'):
        self.transform = transform

        # create empty lists for input / output
        in_MESH = list()
        out_stress = list()
        out_damage = list()

        num_steps = list()

        # loop over directories - different vf, different UC, vp
        keyword = 'L0.05_vf'
        for dir0 in lst_dir0:  #different vf
            VF = re.search(f"{keyword}.*?(\d+\.\d+)", dir0).group(1)

            for dir1 in os.listdir(dir0):  #different UC, vp

                # initial geometry - mesh
                iuc = re.findall(r"\d+", dir1)[0]
                img_name = dir_mesh + '/' + keyword + VF + '/' + 'h0.0002' + '/' + dir1 + '.vtk'
                in_MESH.append(img_name)

                # indices of vtk-steps
                tmp = datasets.utils.list_files(dir0 + '/' + dir1, ".vtk")
                idx = list()
                for tmpi in tmp:
                    idx.append([int(num) for num in re.findall(r"(\d+).vtk", tmpi)])
                idx = np.unique(idx)

                # number of steps
                num_steps.append(len(idx))

                # output: damage
                tmp_stress, tmp_strain, tmp_damage = init_dict_StressStrainDamage()
                for loadstep in idx:
                    registerFileName(tmp_stress, tmp_strain, tmp_damage, dir0 + '/' + dir1 + '/' + LOADprefix, loadstep)
                out_damage.append( tmp_damage )
                out_stress.append( tmp_stress )

        self.in_MESH = in_MESH
        self.out_stress = out_stress
        self.out_damage = out_damage
        self.max_num_steps = max(num_steps)
        self.num_steps = num_steps
        self.outputFields = outputFields

    def __len__(self):
        return len(self.out_damage)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input - mesh
        mesh = vtkFieldReader(self.in_MESH[idx], 'tomo_Volume')

        # output - loop over step
        output = list()
        for step in range(self.num_steps[idx]):

            # damage
            if self.outputFields=='damage' or self.outputFields=='damageANDstress':
                tmp = np.zeros((251, 251, 1))
                for key in self.out_damage[idx]:
                    tmp += vtkFieldReader(self.out_damage[idx][key][step], fieldName=vtk_field_name(key))

             # stress
            if self.outputFields=='stress' or self.outputFields=='damageANDstress':
                if self.outputFields == 'stress':
                    tmp = np.zeros((251, 251, 0))

                for key in self.out_stress[idx]:
                    tmp = np.concatenate( (tmp, vtkFieldReader(self.out_stress[idx][key][step],
                                                               fieldName=vtk_field_name(key))), axis=2 )

            output.append(tmp)

        output = np.stack(output)

        # transform - data augmentation
        if self.transform:

            transform_det = self.transform.to_deterministic()

            # mesh
            mesh = torch.moveaxis(torch.from_numpy(transform_det(image=mesh)), -1, 0)

            # output
            for step in range(self.num_steps[idx]):
                output[step] = transform_det(image=output[step])
            output = torch.moveaxis(torch.from_numpy(output), -1, 1)

        return mesh, output


def pad_collate_video(batch_data):

    (mesh, output) = zip(*batch_data)

    # sequence length
    seq_lens = [x.shape[0] for x in output]

    # min length after padding
    len_min = 10
    nl = max([max(seq_lens), len_min])

    # add a tmp sequence to the end of the batch
    output = list(output)
    _, nc, h, w = output[0].shape
    output.append(torch.zeros(nl, nc, h, w))

    # pack padded sequences to same length
    mesh = pad_sequence(mesh, batch_first=True)
    output = pad_sequence(output, batch_first=True, padding_value=0)

    # remove the tmp sequence
    output = output[:-1]

    # pad with the last-step frame
    seq_lens_max = max(seq_lens)
    for i in range(output.shape[0]):
        for seq in range(seq_lens[i], seq_lens_max):
            output[i,seq] = output[i,seq_lens[i]-1]

    return mesh, output, seq_lens



# =========================================================
# import data - stress/damage at each incremental load step
# =========================================================
class incrementalDataset(Dataset):
    def __init__(self, dir_mesh, lst_dir0=None, LOADprefix='tensX', Isig=1, Ieps=7, transform=None):
        self.transform = transform

        with_step0 = False

        # create empty lists for input / output
        in_stress, in_strain, in_damage = init_dict_StressStrainDamage()
        out_stress, out_strain, out_damage = init_dict_StressStrainDamage()
        in_MESH = list()
        in_LOAD = [list(), list()]
        bad_results = list()

        # loop over directories - different vf, different UC, vp
        for dir0 in lst_dir0:  #different vf

            for dir1 in os.listdir(dir0):  #different UC, vp

                # only unit cells with void are used
                if "vpMIN" not in dir1:
                    continue

                # initial geometry - mesh
                iuc = re.findall(r"\d+", dir1)[0]
                img_name = dir_mesh + '/' + dir1 + '.vtk'
                # mesh = vtkFieldReader(img_name, fieldName='tomo_Volume')

                # indices of vtk-steps
                tmp = datasets.utils.list_files(dir0 + '/' + dir1, ".vtk")
                idx = list()
                for tmpi in tmp:
                    idx.append([int(num) for num in re.findall(r"(\d+).vtk", tmpi)])
                idx = np.unique(idx)

                # stress-strain curves
                file_macro = datasets.utils.list_files(dir0 + '/' + dir1, ".std")[0]
                data_macro = read_macroStressStrain(dir0 + '/' + dir1 + '/' + file_macro)
                idx_peak = np.argmax(data_macro[:, Isig])
                idx_max = idx.max()
                # plot_macro_vtksteps(data_macro, idx, Ieps=7, Isig=1)

                #sanity check
                if idx_max > data_macro.shape[0]:
                    bad_results.append( dir0 + '/' + dir1 )
                    continue

                # register the file names & loading strain (only those corresponding to step10 and stress-drops)
                for loadstep in idx:
                    peakBEFORE = data_macro[:loadstep, Isig].max()

                    if (loadstep == 10):
                        if with_step0:
                            in_LOAD[0].append( 0. )
                            in_LOAD[1].append(data_macro[loadstep - 1, Ieps])
                            in_MESH.append(img_name)
                            registerFileName(in_stress, in_strain, in_damage, zeroVTK=True)
                            registerFileName(out_stress, out_strain, out_damage, dir0+'/'+dir1+'/'+LOADprefix, loadstep)

                        in_LOAD[0].append(data_macro[loadstep - 1, Ieps])
                        in_MESH.append(img_name)
                        registerFileName(in_stress, in_strain, in_damage, dir0+'/'+dir1+'/'+LOADprefix, loadstep)

                    if (data_macro[loadstep-1,Isig] < peakBEFORE) and (loadstep != idx_max):
                        in_LOAD[1].append( data_macro[loadstep-1, Ieps] )
                        in_LOAD[0].append( data_macro[loadstep-1, Ieps] )
                        registerFileName(in_stress, in_strain, in_damage, dir0 + '/' + dir1 + '/' + LOADprefix, loadstep)
                        in_MESH.append( img_name )
                        registerFileName(out_stress, out_strain, out_damage, dir0 + '/' + dir1 + '/' + LOADprefix, loadstep)

                    if (loadstep == idx_max):
                        in_LOAD[1].append( data_macro[loadstep-1, Ieps])
                        registerFileName(out_stress, out_strain, out_damage, dir0 + '/' + dir1 + '/' + LOADprefix, loadstep)

                # # sanity check: if num_input = num_output ?
                # if len(in_stress['sig1']) != len(out_stress['sig1']):
                #     print('Warn: number of input is different from number of output!!!')
                #     break

        self.in_MESH    = in_MESH
        self.in_LOAD    = in_LOAD
        self.in_stress  = in_stress
        self.in_strain  = in_strain
        self.in_damage  = in_damage
        self.out_stress = out_stress
        self.out_strain = out_strain
        self.out_damage = out_damage

    def __len__(self):
        return len(self.in_MESH)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = [list(), list(), np.array([]).reshape(501,501,0), np.zeros((501,501,1))]
        output = [np.array([]).reshape(501,501,0), np.zeros((501,501,1))]

        # input - mesh
        input[0] = vtkFieldReader(self.in_MESH[idx], 'tomo_Volume')

        # input - load
        input[1] = np.array([ self.in_LOAD[0][idx], self.in_LOAD[1][idx] ])

        # input - stress
        if self.in_LOAD[0][idx]==0:
            input[2] = np.zeros((501, 501, 3))
        else:
            for key in self.in_stress:
                # input[2].append( vtkFieldReader(self.in_stress[key][idx], fieldName=vtk_field_name(key)) )
                input[2] = np.concatenate((input[2], vtkFieldReader(self.in_stress[key][idx], fieldName=vtk_field_name(key))), axis=2)
            # input[2] = np.array(input[2])

        # input - damage
        if self.in_LOAD[0][idx]==0:
            input[3] = np.zeros((501, 501, 1))
        else:
            for key in self.in_damage:
                input[3] += vtkFieldReader(self.in_damage[key][idx], fieldName=vtk_field_name(key))

        # output - stress
        for key in self.out_stress:
            # output[0].append( vtkFieldReader(self.out_stress[key][idx], fieldName=vtk_field_name(key)) )
            output[0] = np.concatenate((output[0], vtkFieldReader(self.out_stress[key][idx], fieldName=vtk_field_name(key))), axis=2)
        # output[0] = np.array(output[0])

        # output - damage
        for key in self.out_damage:
            output[1] += vtkFieldReader(self.out_damage[key][idx], fieldName=vtk_field_name(key))

        # transform - data augmentation
        if self.transform:
            transform_det = self.transform.to_deterministic()
            input[0] = torch.from_numpy(np.moveaxis(transform_det(image=input[0]),-1,0))
            input[2] = torch.from_numpy(np.moveaxis(transform_det(image=input[2]),-1,0))
            input[3] = torch.from_numpy(np.moveaxis(transform_det(image=input[3]),-1,0))
            output[0] = torch.from_numpy(np.moveaxis(transform_det(image=output[0]),-1,0))
            output[1] = torch.from_numpy(np.moveaxis(transform_det(image=output[1]),-1,0))

        return input, output






# =========================================================================================
# import data - stress/damage at each incremental load step, considering skipped increments
# =========================================================================================
class incremental2Dataset(Dataset):
    def __init__(self, dir_mesh, lst_dir0=None, LOADprefix='tensX', Isig=1, Ieps=7, transform=None):
        self.transform = transform

        # create empty lists for input / output
        in_stress, in_strain, in_damage = init_dict_StressStrainDamage()
        out_stress, out_strain, out_damage = init_dict_StressStrainDamage()
        in_MESH = list()
        in_LOAD = [list(), list()]
        files_macro = list()
        bad_results = list()

        # loop over directories - different vf, different UC, vp
        for dir0 in lst_dir0:  #different vf

            for dir1 in os.listdir(dir0):  #different UC, vp

                # only unit cells with void are used
                if "vpMIN" not in dir1:
                    continue

                # initial geometry - mesh
                iuc = re.findall(r"\d+", dir1)[0]
                img_name = dir_mesh + '/' + dir1 + '.vtk'
                # mesh = vtkFieldReader(img_name, fieldName='tomo_Volume')

                # indices of vtk-steps
                tmp = datasets.utils.list_files(dir0 + '/' + dir1, ".vtk")
                idx = list()
                for tmpi in tmp:
                    idx.append([int(num) for num in re.findall(r"(\d+).vtk", tmpi)])
                idx = np.unique(idx)

                # stress-strain curves
                file_macro = datasets.utils.list_files(dir0 + '/' + dir1, ".std")[0]
                data_macro = read_macroStressStrain(dir0 + '/' + dir1 + '/' + file_macro)
                idx_peak = np.argmax(data_macro[:, Isig])
                idx_max = idx.max()
                # plot_macro_vtksteps(data_macro, idx, Ieps=7, Isig=1)

                #sanity check
                if idx_max > data_macro.shape[0]:
                    bad_results.append( dir0 + '/' + dir1 )
                    continue

                # register the file names & loading strain
                nsteps = len(idx)
                for i in range(nsteps-1):
                    istart = idx[i]
                    for j in range(i+1,nsteps):
                        iend = idx[j]
                        # print('start {:3d}, end {:3d}'.format(istart, iend))
                        registerFileName(in_stress, in_strain, in_damage, dir0+'/'+dir1+'/'+LOADprefix, istart)
                        registerFileName(out_stress, out_strain, out_damage, dir0+'/'+dir1+'/'+LOADprefix, iend)
                        in_LOAD[0].append(data_macro[istart-1, Ieps])
                        in_LOAD[1].append(data_macro[iend-1, Ieps])
                        in_MESH.append(img_name)
                        files_macro.append(dir0 + '/' + dir1 + '/' + file_macro)

        self.in_MESH    = in_MESH
        self.in_LOAD    = in_LOAD
        self.in_stress  = in_stress
        self.in_strain  = in_strain
        self.in_damage  = in_damage
        self.out_stress = out_stress
        self.out_strain = out_strain
        self.out_damage = out_damage
        self.files_macro = files_macro

    def __len__(self):
        return len(self.in_MESH)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = [list(), list(), np.array([]).reshape(501,501,0), np.zeros((501,501,1))]
        output = [np.array([]).reshape(501,501,0), np.zeros((501,501,1))]

        # input - mesh
        input[0] = vtkFieldReader(self.in_MESH[idx], 'tomo_Volume')

        # input - load
        input[1] = np.array([ self.in_LOAD[0][idx], self.in_LOAD[1][idx] ])

        # input - stress
        if self.in_LOAD[0][idx]==0:
            input[2] = np.zeros((501, 501, 3))
        else:
            for key in self.in_stress:
                input[2] = np.concatenate((input[2], vtkFieldReader(self.in_stress[key][idx], fieldName=vtk_field_name(key))), axis=2)

        # input - damage
        if self.in_LOAD[0][idx]==0:
            input[3] = np.zeros((501, 501, 1))
        else:
            for key in self.in_damage:
                input[3] += vtkFieldReader(self.in_damage[key][idx], fieldName=vtk_field_name(key))

        # output - stress
        for key in self.out_stress:
            # output[0].append( vtkFieldReader(self.out_stress[key][idx], fieldName=vtk_field_name(key)) )
            output[0] = np.concatenate((output[0], vtkFieldReader(self.out_stress[key][idx], fieldName=vtk_field_name(key))), axis=2)
        # output[0] = np.array(output[0])

        # output - damage
        for key in self.out_damage:
            output[1] += vtkFieldReader(self.out_damage[key][idx], fieldName=vtk_field_name(key))

        # transform - data augmentation
        if self.transform:
            transform_det = self.transform.to_deterministic()
            input[0] = torch.from_numpy(np.moveaxis(transform_det(image=input[0]),-1,0))
            input[2] = torch.from_numpy(np.moveaxis(transform_det(image=input[2]),-1,0))
            input[3] = torch.from_numpy(np.moveaxis(transform_det(image=input[3]),-1,0))
            output[0] = torch.from_numpy(np.moveaxis(transform_det(image=output[0]),-1,0))
            output[1] = torch.from_numpy(np.moveaxis(transform_det(image=output[1]),-1,0))

        return input, output






# =======================================================================================================
# import data - stress/damage at each individual load step, for new data with a smaller dimension (L0.05)
# =======================================================================================================

class stepDataset(Dataset):
    def __init__(self, dir_mesh,
                       lst_dir0 = None,
                       LOADprefix = 'Load0.0',
                       transform = None,
                       outputFields = 'damage',
                       varLOAD = False,
                       step = None):

        self.transform = transform

        # create empty lists for input / output
        in_MESH = list()
        in_LOAD = list()

        if outputFields == 'damage':
            _, _, out_field = init_dict_StressStrainDamage()
        elif outputFields == 'stress':
            out_field, _, _ = init_dict_StressStrainDamage()
        elif outputFields == 'strain':
            _, out_field, _ = init_dict_StressStrainDamage()

        # loop over directories - different vf, different UC, vp
        keyword = 'L0.05_vf'
        for dir0 in lst_dir0:  #different vf
            VF = re.search(f"{keyword}.*?(\d+\.\d+)", dir0).group(1)

            for dir1 in os.listdir(dir0):  #different UC, vp

                # initial geometry - mesh
                iuc = re.findall(r"\d+", dir1)[0]
                img_name = dir_mesh + '/' + keyword + VF + '/' + 'h0.0002' + '/' + dir1 + '.vtk'

                # indices of vtk-steps
                tmp = datasets.utils.list_files(dir0 + '/' + dir1, ".vtk")
                idx = list()
                for tmpi in tmp:
                    idx.append([int(num) for num in re.findall(r"(\d+).vtk", tmpi)])
                idx = np.unique(idx)

                # stress-strain curves -> load
                file_macro = datasets.utils.list_files(dir0 + '/' + dir1, ".std")[0]
                data_macro = read_macroStressStrain(dir0 + '/' + dir1 + '/' + file_macro)

                # output: damage / stress
                if varLOAD == True:
                    for loadstep in idx[3:]:
                        if outputFields == 'damage':
                            registerFileName(lst_damage = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                        elif outputFields == 'stress':
                            registerFileName(lst_stress = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                        elif outputFields == 'strain':
                            registerFileName(lst_strain = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)

                        # mesh
                        in_MESH.append(img_name)

                        # load - 6 components: Exx, Eyy, Ezz, Exy, Exz, Ezz
                        in_LOAD.append(data_macro[loadstep-1, 7:13])
                        
                else:
                
                    if step == None:
                        loadstep = idx[-1]
                    else:
                        loadstep = step
                        
                    if outputFields == 'damage':
                        registerFileName(lst_damage = out_field,
                                     fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                     loadstep = loadstep)
                    elif outputFields == 'stress':
                        registerFileName(lst_stress = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                    elif outputFields == 'strain':
                        registerFileName(lst_strain = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)

                    # mesh
                    in_MESH.append(img_name)

                    # load - 6 components: Exx, Eyy, Ezz, Exy, Exz, Ezz
                    in_LOAD.append(data_macro[loadstep-1, 7:13])

        self.in_MESH = in_MESH
        self.in_LOAD = torch.from_numpy(np.stack(in_LOAD))
        self.out_field = out_field
        self.outputFields = outputFields

    def __len__(self):
        return len(self.in_MESH)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input - mesh
        mesh = vtkFieldReader(self.in_MESH[idx], 'tomo_Volume')

        # output - damage / stress
        if self.outputFields=='damage':
            output = np.zeros((251, 251, 1))
            for key in self.out_field:
                output += vtkFieldReader(self.out_field[key][idx], fieldName=vtk_field_name(key))

        elif self.outputFields=='stress':
            output = np.zeros((251, 251, 0))
            for key in self.out_field:
                output = np.concatenate( (output, vtkFieldReader(self.out_field[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )

        elif self.outputFields=='strain':
            output = np.zeros((251, 251, 0))
            for key in self.out_field:
                output = np.concatenate( (output, vtkFieldReader(self.out_field[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )

        # transform - data augmentation
        if self.transform:

            transform_det = self.transform.to_deterministic()
            mesh = transform_det(image=mesh)
            output = transform_det(image=output)

        # to Tensor
        mesh = torch.moveaxis(torch.from_numpy(mesh), -1, 0)
        output = torch.moveaxis(torch.from_numpy(output), -1, 0)

        return mesh, self.in_LOAD[idx], output





# =========================================================================
# import data - stress/strain at elastic step, damage at final step (L0.05)
# =========================================================================

class elas2crkDataset(Dataset):
    def __init__(self, dir_mesh,
                       lst_dir0 = None,
                       LOADprefix = 'Load0.0',
                       transform = None,
                       outputFields = 'damage',
                       varLOAD = False,
                       step = None,
                       PINN = False):

        self.transform = transform

        # create empty lists for input / output
        in_MESH = list()
        in_LOAD = list()
        bad_results = list()

        if outputFields == 'damage':
            _, _, out_field = init_dict_StressStrainDamage()
        elif outputFields == 'stress':
            out_field, _, _ = init_dict_StressStrainDamage()
        elif outputFields == 'strain':
            _, out_field, _ = init_dict_StressStrainDamage()
        elif outputFields == 'elas2crk':
            in_stress, in_strain, out_field = init_dict_StressStrainDamage()
            if PINN == True:
                _, out_strain, _ = init_dict_StressStrainDamage()

        # loop over directories - different vf, different UC, vp
        keyword = 'L0.05_vf'
        for dir0 in lst_dir0:  #different vf
            VF = re.search(f"{keyword}.*?(\d+\.\d+)", dir0).group(1)
            if dir0.find('_CMC/') != -1:
                VF = VF + '_CMC'
                
            for dir1 in os.listdir(dir0):  #different UC, vp

                # initial geometry - mesh
                iuc = re.findall(r"\d+", dir1)[0]
                img_name = dir_mesh + '/' + keyword + VF + '/' + 'h0.0002' + '/' + dir1 + '.vtk'

                # indices of vtk-steps
                tmp = datasets.utils.list_files(dir0 + '/' + dir1, ".vtk")
                idx = list()
                for tmpi in tmp:
                    idx.append([int(num) for num in re.findall(r"(\d+).vtk", tmpi)])
                idx = np.unique(idx)

                # stress-strain curves -> load
                file_macro = datasets.utils.list_files(dir0 + '/' + dir1, ".std")[0]
                data_macro = read_macroStressStrain(dir0 + '/' + dir1 + '/' + file_macro)


                #sanity check
                if idx.max() > data_macro.shape[0]:
                    bad_results.append( dir0 + '/' + dir1 )
                    continue
                    

                # output: damage / stress
                if varLOAD == True:
                    for loadstep in idx[3:]:
                        if outputFields == 'damage':
                            registerFileName(lst_damage = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                        elif outputFields == 'stress':
                            registerFileName(lst_stress = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                        elif outputFields == 'strain':
                            registerFileName(lst_strain = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)

                        # mesh
                        in_MESH.append(img_name)

                        # load - 6 components: Exx, Eyy, Ezz, Exy, Exz, Ezz
                        in_LOAD.append(data_macro[loadstep-1, 7:13])
                        
                else:
                
                    if step == None:
                        loadstep = idx[-1]
                    else:
                        loadstep = step
                        
                    if outputFields == 'damage':
                        registerFileName(lst_damage = out_field,
                                     fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                     loadstep = loadstep)
                    elif outputFields == 'stress':
                        registerFileName(lst_stress = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                    elif outputFields == 'strain':
                        registerFileName(lst_strain = out_field,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                    elif outputFields == 'elas2crk':
                        registerFileName(lst_stress = in_stress,
                                     fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                     loadstep = idx[0])
                        registerFileName(lst_strain = in_strain,
                                     fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                     loadstep = idx[0])
                        registerFileName(lst_damage = out_field,
                                     fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                     loadstep = loadstep)
                        if PINN == True:
                            registerFileName(lst_strain = out_strain,
                                         fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                                         loadstep = loadstep)
                            

                    # mesh
                    in_MESH.append(img_name)

                    # load - 6 components: Exx, Eyy, Ezz, Exy, Exz, Ezz
                    in_LOAD.append(data_macro[loadstep-1, 7:13])

        self.in_MESH = in_MESH
        self.in_LOAD = torch.from_numpy(np.stack(in_LOAD))
        self.out_field = out_field
        self.outputFields = outputFields
        self.PINN = PINN
        if outputFields == 'elas2crk':
            self.in_stress = in_stress
            self.in_strain = in_strain
            if PINN == True:
                self.out_strain = out_strain
            
        self.badresults = bad_results

    def __len__(self):
        return len(self.in_MESH)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input - mesh
        mesh = vtkFieldReader(self.in_MESH[idx], 'tomo_Volume')

        # output - damage / stress
        if self.outputFields=='damage':
            output = np.zeros((251, 251, 1))
            for key in self.out_field:
                output += vtkFieldReader(self.out_field[key][idx], fieldName=vtk_field_name(key))

        elif self.outputFields=='stress':
            output = np.zeros((251, 251, 0))
            for key in self.out_field:
                output = np.concatenate( (output, vtkFieldReader(self.out_field[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )

        elif self.outputFields=='strain':
            output = np.zeros((251, 251, 0))
            for key in self.out_field:
                output = np.concatenate( (output, vtkFieldReader(self.out_field[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )
                                                                 
        elif self.outputFields=='elas2crk':
            stress = np.zeros((251, 251, 0))
            for key in self.in_stress:
                stress = np.concatenate( (stress, vtkFieldReader(self.in_stress[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )
            strain = np.zeros((251, 251, 0))
            for key in self.in_strain:
                strain = np.concatenate( (strain, vtkFieldReader(self.in_strain[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )
            energy = np.zeros((251, 251, 1))
            energy[:,:,0] = strain[:,:,0] * stress[:,:,0] + strain[:,:,1] * stress[:,:,1] + 2* strain[:,:,2] * stress[:,:,2]
            energy = energy/energy.max()

            output = np.zeros((251, 251, 1))
            for key in self.out_field:
                output += vtkFieldReader(self.out_field[key][idx], fieldName=vtk_field_name(key))
                
            if self.PINN == True:
                for key in self.out_strain:
                    output = np.concatenate( (output, vtkFieldReader(self.out_strain[key][idx],
                                                                 fieldName=vtk_field_name(key))), axis=2 )
        

        # transform - data augmentation
        if self.transform:

            transform_det = self.transform.to_deterministic()
            mesh = transform_det(image=mesh)
            output = transform_det(image=output)
            
            if self.outputFields=='elas2crk':
                energy = transform_det(image=energy)

        # to Tensor
        mesh = torch.moveaxis(torch.from_numpy(mesh), -1, 0)
        output = torch.moveaxis(torch.from_numpy(output), -1, 0)
        if self.outputFields=='elas2crk':
            energy = torch.moveaxis(torch.from_numpy(energy), -1, 0)

        return mesh, energy, self.in_LOAD[idx], output









# ================================================
# import data - macro stress/strain curves (L0.05)
# ================================================

class macrocurveDataset(Dataset):
    def __init__(self, dir_mesh,
                       lst_dir0 = None,
                       LOADprefix = 'Load0.0',
                       transform = None,
                       seq_len = 1000
                       ):

        self.transform = transform

        # create empty lists for input / output
        in_MESH = list()
        out_CURVE = list()

        # loop over directories - different vf, different UC, vp
        keyword = 'L0.05_vf'
        for dir0 in lst_dir0:  #different vf
            VF = re.search(f"{keyword}.*?(\d+\.\d+)", dir0).group(1)

            for dir1 in os.listdir(dir0):  #different UC, vp

                # initial geometry - mesh
                iuc = re.findall(r"\d+", dir1)[0]
                img_name = dir_mesh + '/' + keyword + VF + '/' + 'h0.0002' + '/' + dir1 + '.vtk'
                in_MESH.append(img_name)

                # stress-strain curves
                file_macro = datasets.utils.list_files(dir0 + '/' + dir1, ".std")[0]
                data_macro = read_macroStressStrain(dir0 + '/' + dir1 + '/' + file_macro)

                # unify the length
                eps = np.linspace(0, data_macro[-1,7], seq_len)
                sig = np.interp(eps, data_macro[:,7], data_macro[:,1])
                
                out_CURVE.append(np.array([eps, sig]))


        self.in_MESH = in_MESH
        self.out_CURVE = out_CURVE
        

    def __len__(self):
        return len(self.in_MESH)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input - mesh
        mesh = vtkFieldReader(self.in_MESH[idx], 'tomo_Volume')


        # transform - data augmentation
        if self.transform:

            transform_det = self.transform.to_deterministic()
            mesh = transform_det(image=mesh)

        # to Tensor
        mesh = torch.moveaxis(torch.from_numpy(mesh), -1, 0)
        curve = torch.from_numpy(self.out_CURVE[idx])

        return mesh, curve




















# ======================================
# functions for saving and loading model
# ======================================

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()




# ==============================
# learning curve post-processing
# ==============================
# https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py


import traceback

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

mesh_paths = (r"D:\data\mesh\L0.05_vf0.3\h0.0002",
              r"D:\data\mesh\L0.05_vf0.5\h0.0002")

response_paths = (r"D:\data\L0.05_vf0.3\h0.0002",
               r"D:\data\L0.05_vf0.5\h0.0002")

def load_damage_field_prediction(lst_dir0, dir_mesh,
                              n_train, n_tests,
                              batch_size, test_batch_sizes,
                              test_resolutions=[251],
                              train_resolution=251,
                              grid_boundaries=[[0, 1], [0, 1]],
                              positional_encoding=True,
                              encode_input=True,
                              encode_output=True,
                              encoding='channel-wise',
                              seq_len=1000):
    # create empty lists for input / output
    _, _, out_field = init_dict_StressStrainDamage()

    # loop over directories - different vf, different UC, vp
    keyword = 'L0.05_vf'
    LOADprefix = 'Load0.0'
    loadstep = 10
    x = list()
    y = list()
    for mesh_path, response_path in zip(mesh_paths, response_paths):

        #loop over folders in mesh directory to set a same order and have them correspond
        for folder in os.listdir(response_path):
            #if folder == '.DS_Store':
                #continue
            # initial geometry from mesh
            img_names = glob.glob(f'{mesh_path}/{folder}.vtk')
            # appending geometries to the input list
            [x.append(vtkFieldReader(name, 'tomo_Volume')) for name in img_names]


            # output: stress (filename)
            registerFileName(lst_damage=out_field,
                             fprefix=f'{response_path}/{folder}/{LOADprefix}',
                             loadstep=loadstep)

    # output: damage (data)
    nsamples = len(x)
    for i in range(nsamples):
        try:
            output = np.zeros((251, 251, 0))
            for key in out_field:
                output = np.concatenate((output, vtkFieldReader(out_field[key][i],
                                                            fieldName=vtk_field_name(key))), axis=2)
            y.append(output)
        except:
            continue
    ##
    x = np.expand_dims(np.array(x), 1)  # BxCxWxHxD
    y = np.expand_dims(np.moveaxis(np.array(y), -1, 1), -1)  # BxCxWxHxD

    idx_train = np.random.choice(len(x)-1, n_train)  # random shuffle
    x_train = torch.from_numpy(x[idx_train, :, :, :, 0]).type(torch.float32).clone()
    y_train = torch.from_numpy(y[idx_train, :, :, :, 0]).type(torch.float32).clone()

    #
    test_batch_size = test_batch_sizes[0]
    test_resolution = test_resolutions[0]

    n_test = n_tests[0]  # currently, only 1 resolution possible
    idx_test = np.random.choice(np.setdiff1d(np.arange(len(x)), idx_train), n_test)
    x_test = torch.from_numpy(x[idx_test, :, :, :, 0]).type(torch.float32).clone()
    y_test = torch.from_numpy(y[idx_test, :, :, :, 0]).type(torch.float32).clone()

    ## input encoding
    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(x_train, reduce_dim=reduce_dims)
        x_train = input_encoder.encode(x_train)
        x_test = input_encoder.encode(x_test.contiguous())
    else:
        input_encoder = None

    ## output encoding
    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(y_train, reduce_dim=reduce_dims)
        y_train = output_encoder.encode(y_train)
    else:
        output_encoder = None

    ## training dataset
    train_db = TensorDataset(x_train, y_train,
                             transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
    train_loader = torch.utils.data.DataLoader(train_db,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True, persistent_workers=False)

    ## test dataset
    test_db = TensorDataset(x_test, y_test,
                            transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, persistent_workers=False)

    test_loaders = {train_resolution: test_loader}
    test_loaders[test_resolution] = test_loader  # currently, only 1 resolution is possible

    return train_loader, test_loaders, output_encoder

