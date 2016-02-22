Background
==========
The 'Destrieux' cortical atlas is based on a parcellation scheme
that first divided the cortex into gyral and sulcal regions,
the limit between both being given by the curvature value of the surface.
A gyrus only includes the cortex visible on the pial view,
the hidden cortex (banks of sulci) are marked sulcus.
The result is a complete labeling of cortical sulci and gyri.

Files in Folder (Excluding README)
==================================
1) rh_destrieux2009_rois.nii.gz
2) lh_destrieux2009_rois.nii.gz
3) destrieux2009_rois_labels.csv

Descriptions of files
=====================
1) rh_destrieux2009_rois.nii.gz is a right hemisphere volume consisting
of 76 rois projected into a 2mm isotropic MNI152 space.
The volume is obtained from FreeSurfer's fsaverage template, by transforming
the annotation to a volume with mri_label2vol command as follows :
mri_label2vol --subject fsaverage --hemi rh --annot rh.aparc.a2009s.annot
--reg reg.2mm.dat --temp brain.mgz --fillthresh 0. --proj frac 0 1.5 .1
--o rh_destrieux2009_rois.nii.gz

2) lh_destrieux2009_rois.nii.gz is the left hemisphere counterpart volume.

3) destrieux2009_rois_labels.csv is the file that contains the rois labels.

References
==========
Fischl, Bruce, et al. "Automatically parcellating the human cerebral cortex."
Cerebral cortex 14.1 (2004): 11-22.

Destrieux, C., et al. "A sulcal depth-based anatomical parcellation
of the cerebral cortex." NeuroImage 47 (2009): S151.
