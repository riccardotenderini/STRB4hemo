#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:08:34 2021
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import os
import numpy as np
import h5py
import re


def create_dir(name):
    """Create a directory at the given path

    :param name: path of the directory to be created, if not already existing
    :type name: str

    """
    if not os.path.exists(name):
        os.makedirs(name)
    return


def get_full_subdirectories(rootdir):
    subdirs = []
    for file in sorted(os.listdir(rootdir)):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            subdirs.append(d)
    return subdirs


def read_mesh(filename):
    _, file_extension = os.path.splitext(filename)
    if file_extension != '.mesh':
        raise ValueError(f"Invalid file extension {file_extension}! Only .mesh files are supported.")

    mesh = dict()

    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    for cnt_line, line in enumerate(lines):
        if line in {'\n', '', ' '}:
            continue

        elif 'MeshVersionFormatted' in line:
            mesh['version'] = [int(s) for s in line.split() if s.isdigit()][0]

        elif 'Dimension' in line:
            mesh['dimension'] = [int(s) for s in line.split() if s.isdigit()][0]
            if mesh['dimension'] != 3:
                raise ValueError("Only 3D meshes can be read!")

        elif 'Vertices' in line:
            vertices_line = cnt_line

            cnt_line += 1
            line = lines[cnt_line]

            mesh['num_vertices'] = int(line)
            mesh['vertices'] = np.zeros((mesh['num_vertices'], mesh['dimension']), dtype=np.float32)
            mesh['vertices-flags'] = np.zeros(mesh['num_vertices'], dtype=np.int32)

            for cnt_line in range(vertices_line + 2, vertices_line + 2 + mesh['num_vertices']):
                line = lines[cnt_line]
                idx_row = cnt_line - vertices_line - 2
                vals = [float(s) for s in line.split()]
                if len(vals) != mesh['dimension'] + 1:
                    raise ValueError(
                        f"Line {cnt_line} has {len(vals)} values, while {mesh['dimension'] + 1} are expected!")

                mesh['vertices'][idx_row, :] = vals[:-1]
                mesh['vertices-flags'][idx_row] = int(vals[-1])

        elif 'Triangles' in line:
            triangles_line = cnt_line

            cnt_line += 1
            line = lines[cnt_line]

            mesh['num_triangles'] = int(line)
            mesh['triangles'] = np.zeros((mesh['num_triangles'], 3), dtype=np.int32)
            mesh['triangles-flags'] = np.zeros(mesh['num_triangles'], dtype=np.int32)

            for cnt_line in range(triangles_line + 2, triangles_line + 2 + mesh['num_triangles']):
                line = lines[cnt_line]
                idx_row = cnt_line - triangles_line - 2
                vals = [int(s) for s in line.split()]
                if len(vals) != 4:
                    raise ValueError(f"Line {cnt_line} has {len(vals)} values, while 4 are expected!")

                mesh['triangles'][idx_row, :] = vals[:-1]
                mesh['triangles-flags'][idx_row] = vals[-1]

        elif 'Tetrahedra' in line:
            tetrahedra_line = cnt_line

            cnt_line += 1
            line = lines[cnt_line]

            mesh['num_tetrahedra'] = int(line)
            mesh['tetrahedra'] = np.zeros((mesh['num_tetrahedra'], 4), dtype=np.int32)
            mesh['tetrahedra-flags'] = np.zeros(mesh['num_tetrahedra'], dtype=np.int32)

            for cnt_line in range(tetrahedra_line + 2, tetrahedra_line + 2 + mesh['num_tetrahedra']):
                line = lines[cnt_line]
                idx_row = cnt_line - tetrahedra_line - 2
                vals = [int(s) for s in line.split()]
                if len(vals) != 5:
                    raise ValueError(f"Line {cnt_line} has {len(vals)} values, while 5 are expected!")

                mesh['tetrahedra'][idx_row, :] = vals[:-1]
                mesh['tetrahedra-flags'][idx_row] = vals[-1]

    return mesh


def read_field_from_h5(hdf5_file, field):
    with h5py.File(hdf5_file, "r") as hdf5:
        field_name = lambda _cnt: f"{field}.{_cnt:05d}/Values"

        cnt = 0
        field_values = []
        while field_name(cnt) in hdf5:
            field_values.append(hdf5[field_name(cnt)][:].T.flatten())
            cnt += 1

        field_values = np.array(field_values)

    return field_values


def write_field_to_h5(hdf5_file, field, field_values, field_dim=1, extend=True):
    with (h5py.File(hdf5_file, "r+") as hdf5):
        field_name = lambda _cnt: f"{field}.{_cnt:05d}/Values"

        cnt = 0
        while field_name(cnt) in hdf5:
            hdf5[field_name(cnt)][:] = field_values[:, cnt].reshape(field_dim, -1).T
            cnt += 1

    if cnt < field_values.shape[1] and extend:
        __extend_field_in_h5(hdf5_file, cnt, field, field_values, field_dim=field_dim)

    return


def __extend_field_in_h5(hdf5_file, old_t, field, field_values, field_dim=1):
    with (h5py.File(hdf5_file, "r+") as hdf5):
        points_x = hdf5[f"PointsX.00000/Values"][:]
        points_y = hdf5[f"PointsY.00000/Values"][:]
        points_z = hdf5[f"PointsZ.00000/Values"][:]

    with h5py.File(hdf5_file, "a") as h5f:

        for t in range(old_t, field_values.shape[1]):
            t_str = f"{t:05d}"

            if f"PointsX.{t_str}/Values" not in h5f:
                h5f.create_dataset(f"PointsX.{t_str}/Values", data=points_x)
                h5f.create_dataset(f"PointsY.{t_str}/Values", data=points_y)
                h5f.create_dataset(f"PointsZ.{t_str}/Values", data=points_z)

            if f"{field}.{t_str}/Values" not in h5f:
                h5f.create_dataset(f"{field}.{t_str}/Values", data=field_values[:, t].reshape(field_dim, -1).T)

    return


def update_xmf_file(xmf_file_old, xmf_file_new, old_t, new_t, dt):

    with open(xmf_file_old, "r") as file:
        xmf_contents = file.read()

    # Extract the first timestep block based on the comment
    match = re.search(r'(<!-- Time .*? Iteration 00000 -->.*?</Grid>)', xmf_contents, re.DOTALL)
    if not match:
        raise ValueError("Could not find the first timestep block in the XMF file.")

    first_timestep_block = match.group(1)  # The full block for timestep 00000

    # Find the first timestep number
    first_time_match = re.search(r'Value="([\d.]+)"', first_timestep_block)
    if not first_time_match:
        raise ValueError("Could not find the first timestep value in the XMF file.")

    first_time_value = float(first_time_match.group(1))  # First timestep value

    # Generate new <Grid> blocks for timesteps
    new_entries = []
    for t in range(old_t, new_t):  # Adding timesteps
        new_time_value = first_time_value + (t * dt)  # Estimate time value
        t_str = f"{t:05d}"  # Format as '00016', '00017', etc.

        # Replace occurrences of '00000' with the new timestep number
        new_block = re.sub(r'00000', t_str, first_timestep_block)
        new_block = re.sub(f"{first_time_value}", f"{new_time_value:.5f}", new_block)

        new_entries.append(new_block)

    # Find all occurrences of <Grid>
    grid_positions = [m.start() for m in re.finditer(r"</Grid>", xmf_contents)]

    if len(grid_positions) < 2:
        raise ValueError("Not enough </Grid> elements found to insert before the second-to-last.")

    # Find the second-to-last Grid position
    insert_position = grid_positions[-2]  # Last but one

    # Insert new content before the second-to-last Grid
    updated_xmf_contents = (
            xmf_contents[:insert_position + 7] + "\n".join(new_entries) + xmf_contents[insert_position + 7:]
    )

    # Write the updated XMF file
    with open(xmf_file_new, "w") as file:
        file.write(updated_xmf_contents)

    return


def add_field_to_xmf(xmf_file, field_name):

    with open(xmf_file, "r") as file:
        xmf_contents = file.read()

    # Find all velocity blocks per timestep
    velocity_blocks = list(re.finditer(r'(<Attribute[^>]+Name="velocity"[\s\S]*?</Attribute>)',
                                       xmf_contents, re.DOTALL))

    if not velocity_blocks:
        raise ValueError("No 'velocity' attribute blocks found in the XMF file.")

    # Modify each timestep by duplicating the velocity block as WSS
    updated_xmf_contents = xmf_contents

    for velocity in reversed(velocity_blocks):  # Process in reverse to avoid shifting indices
        velocity_block = velocity.group(1)

        # Extract the timestep from the DataStructure path
        match_timestep = re.search(r'/velocity.(\d{5})/Values', velocity_block)
        if not match_timestep:
            continue  # Skip if no matching timestep is found

        t_str = match_timestep.group(1)  # Extracted timestep (zero-padded)

        # Create the WSS block by replacing "velocity" with the new field name
        new_block = re.sub(r'Name="velocity"', f'Name="{field_name}"', velocity_block)
        new_block = re.sub(r'/velocity.(\d{5})/Values', f'/{field_name}.{t_str}/Values', new_block)

        # Insert the new block after the Velocity block
        insert_position = velocity.end()
        updated_xmf_contents = (updated_xmf_contents[:insert_position] + "\n\n\t" +
                                new_block + updated_xmf_contents[insert_position:])

    # Save the modified XMF file
    with open(xmf_file, "w") as file:
        file.write(updated_xmf_contents)

    return


__all__ = [
    "create_dir",
    "get_full_subdirectories",
    "read_mesh",
    "read_field_from_h5",
    "write_field_to_h5",
    "update_xmf_file",
    "add_field_to_xmf"
]
