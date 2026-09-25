"""Which steerable properties exist, where their descriptors are, and what each hashes.

**This file is deliberately NOT a member of any fingerprint group, and that is the point.** It
is the one file a fourth property has to edit, so hashing it into the shared groups would
re-key every existing checkpoint for the arrival of a property that executes none of their
code -- the exact defect the registry exists to end.

It is not fail-open, for two reasons spelled out in
:mod:`sae_bioemu.steering.property_registry`:

* the group memberships below enter every digest BY VALUE (``_code_fingerprints`` hashes the
  ``{group: {member: content digest}}`` map itself), so this file cannot make a run hash less
  without moving that run's digest; and
* ``property_registry.resolve`` refuses a descriptor whose own source file is not among its
  property's training-group members, so this file cannot point a name at different behaviour
  while the digest stands still.

Everything here is therefore a name, the spelling of a config block, or a file list. Not one
scientific quantity lives in this file: the dose set, the target origin, the callback versions,
the two-state declaration, the objective class and the refused weights are all in the
property's own descriptor, hashed in the property's own group.

ADDING A PROPERTY
-----------------

1. Write its descriptor in its own package and list that file in one of its own training
   groups (``resolve`` enforces the second half).
2. Add a :class:`~sae_bioemu.steering.property_registry.PropertyRegistration` below.

No shared file is touched, so no existing checkpoint moves.
"""

from __future__ import annotations

from pathlib import Path

from sae_bioemu.steering.property_registry import PropertyRegistration

__all__ = ["STEERABLE_PROPERTY_REGISTRATIONS"]

_PACKAGE = Path(__file__).resolve().parents[1]
_STEERING = _PACKAGE / "steering"
_BIOEMU = _PACKAGE / "bioemu"
_CRYPTIC = _PACKAGE / "cryptic_steering"
_CRYPTIC_RMSD = _PACKAGE / "cryptic_rmsd"
_KINASE = _PACKAGE / "kinase_dfg"
_KINASE_RMSD = _PACKAGE / "kinase_rmsd"
_DOMAIN_MOTION = _PACKAGE / "domain_motion"

#: The generation-shared sources BOTH two-state properties execute and radius of gyration does
#: not. ``cryptic_generation.py`` reconstructs N, CA, C and the virtual C-beta from the
#: denoiser's per-residue frames, so it decides which ATOMS the progress coordinate is measured
#: between; a change to it changes every ``the progress coordinate`` in both programmes and changed no digest before
#: this group existed. It is not in ``generation_shared`` because a radius-of-gyration run never
#: calls it -- ``measurement_atom_set`` is ``None`` there and nothing but C-alpha is ever read
#: -- and it is not in ``cryptic_generation_shared`` because it belongs to neither package.
#:
#: Declared identically by both registrations. ``scientific_code_paths`` merges them and refuses
#: two declarations of one member as different files, so the duplication cannot drift.
_TWO_STATE_GENERATION_SHARED = {
    "bioemu_cryptic_generation": _BIOEMU / "cryptic_generation.py",
}

#: The differentiable aligned-RMSD kernel, shared by every property whose objective measures a
#: direct Angstrom distance to a reference STRUCTURE rather than a progress coordinate.
#: ``aligned_rmsd.py`` owns the Kabsch superposition, the reflection correction and the detached
#: rotation whose exactness the envelope theorem supplies, so it decides what "this structure is
#: 1.3 Angstrom from that one" MEANS and which direction the gradient of that statement points.
#:
#: It is its own group rather than a member of an existing one for the reason the registry
#: exists. ``training_only`` or ``generation_shared`` would re-key radius of gyration and kinase,
#: neither of which executes a line of it. ``cryptic_generation_shared`` would re-key kinase for
#: the same reason. ``two_state_generation_shared`` would be a lie about what the group means: it
#: names the code both PROGRESS-COORDINATE properties execute, and this kernel is executed by
#: neither of them today.
#:
#: Declared identically by every registration that names it: the cryptic and the domain-motion
#: ones carry the same two lines, byte for byte. ``scientific_code_paths`` merges the declarations
#: and refuses two declarations of one member as different files, so the duplication cannot drift
#: -- that refusal is what makes sharing a group safe rather than a convention.
_ALIGNED_RMSD_SHARED = {
    "steering_aligned_rmsd": _STEERING / "aligned_rmsd.py",
}

STEERABLE_PROPERTY_REGISTRATIONS: dict[str, PropertyRegistration] = {
    "radius_of_gyration": PropertyRegistration(
        name="radius_of_gyration",
        descriptor_module="sae_bioemu.steering.rg_property",
        descriptor_attribute="RADIUS_OF_GYRATION",
        # No arm: the target is built from the reference table alone.
        arm_config_key=None,
        code_group_members={
            # One member, and no ``rg_training_only`` or ``rg_evaluation_only`` beside it.
            # Radius of gyration adds no term to the shared loss and no scorer of its own --
            # ``bioemu/steering_evaluation.py`` is in the shared ``evaluation_only`` group --
            # so an empty group would be a declaration with nothing to declare.
            "rg_generation_shared": {"rg_property": _STEERING / "rg_property.py"},
        },
        code_group_composition={
            "training": ("training_only", "generation_shared", "rg_generation_shared"),
            "evaluation": ("evaluation_only",),
        },
        # No off-path column and no private runtime key. Radius of gyration adds no term to the
        # shared loss, so its journal row is the base schema plus whatever the SHARED weights
        # gate, and an empty declaration here is what keeps it byte-identical.
    ),
    "pocket_progress_coordinate": PropertyRegistration(
        name="pocket_progress_coordinate",
        descriptor_module="sae_bioemu.cryptic_steering.property",
        descriptor_attribute="CRYPTIC_PROPERTY",
        arm_config_key="cryptic_arm",
        code_group_members={
            # ``loss.py`` defines lambda_perp, the context-rigidity hinge and the total the
            # adapter is trained to descend; ``direct_metric.py`` defines
            # what the ``lambda_m`` term IN that total means -- the deviation form, the hinge
            # shape, the per-system normalisation and which deposited conformer a level names.
            #
            # Training-only, and the group choice is the whole point of the exercise. The
            # direct-metric term changes what the adapter is trained to descend and nothing else:
            # no evaluation computes a loss, so a change here cannot move a generated structure or
            # a scored number, and it must not be able to invalidate an evaluation. Putting it in
            # ``cryptic_generation_shared`` instead would have re-keyed every KINASE checkpoint --
            # that group is named by the kinase composition, because a kinase run executes the
            # cryptic pair geometry -- for a term no kinase arm can configure.
            "cryptic_training_only": {
                "cryptic_direct_metric": _CRYPTIC / "direct_metric.py",
                "cryptic_loss": _CRYPTIC / "loss.py",
            },
            # The sources that decide what a cryptic structure, or the quantity read off it, IS
            # in both paths:
            #
            #   pocket          computes delta_d_star and owns TRANSITION_CLASSES: it turns
            #                   coordinates into the distance space the coordinate lives in.
            #   objective       DEFINES the progress coordinate and zeta_pocket from that distance change.
            #   artifact        supplies the pair set, the weights, the transition vector and
            #                   the residue correspondence, so it decides which residues the progress coordinate is
            #                   measured over.
            #   arm             resolves the configuration, runs the closing gates and is the
            #                   per-batch call site producing the loss inputs. It defines the
            #                   TARGET, which is why it is not training-only: that would let an
            #                   evaluation score a checkpoint against a differently-resolved
            #                   arm, the fail-open direction.
            #   site            builds the per-residue channel s_i arms A and B condition on,
            #                   including the smoothing kernel and its width.
            #   apo_reference   under t = the progress coordinate_state(o) - the progress coordinate_reporting(control) these coordinates
            #                   are the ORIGIN every two-state target is measured from.
            #   property        this property's descriptor: the dose set, the frame, the
            #                   callback versions and the objective class.
            #
            # ``calibration`` and ``stage1`` JOINED this group with the registry work, and they
            # close a fail-OPEN rather than adding caution. ``arm.py`` imports
            # ``calibration.REPORTING_REFERENCE_KIND`` and ``stage1.context_setpoint_table`` at
            # MODULE scope, so both are executed by every cryptic arm resolution: the first
            # names the frame the reported coordinate is anchored in and the second reads the
            # per-protein c0 the rigidity hinge is measured against. The standing argument for
            # leaving stage 1 out -- "it writes artifacts that carry their own digests" -- is
            # true of the WRITER and false of the reader the arm calls.
            "cryptic_generation_shared": {
                "cryptic_apo_reference": _CRYPTIC / "apo_reference.py",
                "cryptic_arm": _CRYPTIC / "arm.py",
                "cryptic_artifact": _CRYPTIC / "artifact.py",
                "cryptic_calibration": _CRYPTIC / "calibration.py",
                "cryptic_objective": _CRYPTIC / "objective.py",
                "cryptic_pocket": _CRYPTIC / "pocket.py",
                "cryptic_property": _CRYPTIC / "property.py",
                "cryptic_site": _CRYPTIC / "site.py",
                "cryptic_stage1": _CRYPTIC / "stage1.py",
            },
            "two_state_generation_shared": dict(_TWO_STATE_GENERATION_SHARED),
            "aligned_rmsd_shared": dict(_ALIGNED_RMSD_SHARED),
            # ``steering_evaluation_cryptic`` runs the held-out evaluation, ``scoring`` defines
            # the primary and secondary metrics, and ``structure_gates`` produces the gate
            # verdict every the progress coordinate is required to be reported beside. None of the three is read by
            # the training loop, so changing any of them re-keys an evaluation and retires no
            # checkpoint.
            "cryptic_evaluation_only": {
                "bioemu_steering_evaluation_cryptic": _BIOEMU / "steering_evaluation_cryptic.py",
                "cryptic_scoring": _CRYPTIC / "scoring.py",
                "cryptic_structure_gates": _CRYPTIC / "structure_gates.py",
            },
        },
        code_group_composition={
            "training": (
                "training_only",
                "generation_shared",
                "two_state_generation_shared",
                "aligned_rmsd_shared",
                "cryptic_training_only",
                "cryptic_generation_shared",
            ),
            "evaluation": ("evaluation_only", "cryptic_evaluation_only"),
        },
        # The persisted off-path columns this objective reports, each with the weight that gates
        # it. ``perp_loss`` and ``rigidity_loss`` are the D-28 split and have been written since
        # the term existed; the two metric columns are new and BOTH ride ``lambda_m``, so a
        # cryptic arm that does not price the direct metric writes the byte-identical row it
        # wrote before the term existed.
        #
        # Two columns rather than one, deliberately. ``metric_loss`` is the dimensionless hinge
        # the loss actually descends and ``metric_deviation_angstrom`` is the measured deviation
        # in the benchmark's own unit. A hinge of zero means "inside tau_p" and cannot tell a
        # draw at 2.9 Angstrom from one at 0.4, so an arm journalling only the hinge would be
        # unreadable exactly where it was working.
        journal_columns={
            "perp_loss": "lambda_perp",
            "rigidity_loss": "lambda_rigid",
            "metric_loss": "lambda_m",
            "metric_deviation_angstrom": "lambda_m",
        },
        # D-86: the realised-alignment MONITOR, on training rows, PRICED NOWHERE. A name and
        # nothing else, because a monitor has no gating weight -- that is what makes it a monitor
        # rather than a term, and it is why it cannot ride ``perp_loss`` above.
        #
        # ``zeta_pocket^2``, dimensionless, so ``|cos| = |achieved|/sqrt(achieved^2 + this)`` is
        # the statistic D-76 section 3 measured at 0.165 against an oracle 0.740. That measurement
        # was only possible because ``cryptic_arm1/ddp2`` priced ``lambda_perp``; the DIRECT-METRIC
        # arm does not, and D-80 section 4 names it as the one candidate that changes the direction
        # asked for at ``u ~= 0``. So the arm that could falsify D-77's ceiling was the one arm with
        # no column that could.
        monitor_columns=("pocket_off_axis_ratio_squared",),
        # The direct-metric term's weight and its threshold. They live in the ``bioemu_pytorch``
        # runtime block with every other weight rather than in the arm config, and they are
        # declared HERE rather than added to the workflow's shared optional key set: the workflow
        # admits the union and refuses a key owned by another property, so ``lambda_m`` on a
        # kinase config fails by name instead of being dropped by a constructor that cannot take
        # it. ``metric_tau_angstrom`` is not a weight, and it travels with the weight for the
        # reason ``context_setpoint_percentile`` does -- two runs at one ``lambda_m`` and two taus
        # are two objectives (D-63), so the threshold has to be configurable wherever the weight
        # is and recorded in the same identity.
        optional_runtime_keys=("lambda_m", "metric_tau_angstrom"),
        # The ONE non-default spelling this property's selection hook can return, admitted here so
        # that the completion gate -- which holds a persisted ``adapter_config`` and no objective
        # -- can validate a checkpoint that recorded it. Every SHIPPED cryptic arm runs at
        # ``lambda_m = 0`` and selects on the shared ``target_mae_sigma``, so this widens what is
        # expressible and changes no arm on disk.
        selection_metric_names=("metric_deviation_angstrom_mean",),
    ),
    "dfg_progress_coordinate": PropertyRegistration(
        name="dfg_progress_coordinate",
        descriptor_module="sae_bioemu.kinase_dfg.property",
        descriptor_attribute="KINASE_PROPERTY",
        arm_config_key="kinase_arm",
        code_group_members={
            "kinase_training_only": {"kinase_loss": _KINASE / "loss.py"},
            # The sources that decide what a kinase structure, or the quantity read off it, IS
            # in both paths. ``objective`` DEFINES the progress coordinate and zeta_target; ``arm`` resolves the
            # configuration, runs the closing gates and is the per-batch call site producing the
            # tensors the loss consumes; ``artifact`` decides which residues the progress coordinate is measured
            # over, which atom set it is read in and where the motif window sits;
            # ``dihedral_gradient`` and ``the progress coordinate_gradient`` place the atoms and embed the motif;
            # ``kincore`` holds the centroids d_K is measured to; ``property`` is this
            # property's descriptor.
            #
            # ``dihedral_coordinate``, ``distance_coordinate``, ``geometry`` and ``motif_pairs``
            # are declared HERE rather than left out, and the inclusion is deliberate: each
            # defines a coordinate, a motif window or a torsion the objective could come to
            # price, and adding a term that reads one must not also change this group's
            # membership. They are the coordinate machinery, not the probes -- ``headroom``,
            # ``panel``, ``site_sweep``, ``paired_gap``, ``gap_uncertainty``,
            # ``label_coupling``, ``off_path_direction`` and ``kincore_release`` write or
            # analyse artifacts whose own digests already reach the arm identity, and hashing
            # them would retire checkpoints for a change to a writer they never ran.
            "kinase_generation_shared": {
                "kinase_arm": _KINASE / "arm.py",
                "kinase_artifact": _KINASE / "artifact.py",
                "kinase_dihedral_coordinate": _KINASE / "dihedral_coordinate.py",
                "kinase_dihedral_gradient": _KINASE / "dihedral_gradient.py",
                "kinase_distance_coordinate": _KINASE / "distance_coordinate.py",
                "kinase_geometry": _KINASE / "geometry.py",
                "kinase_kincore": _KINASE / "kincore.py",
                "kinase_motif_pairs": _KINASE / "motif_pairs.py",
                "kinase_objective": _KINASE / "objective.py",
                "kinase_property": _KINASE / "property.py",
                "kinase_the progress coordinate_gradient": _KINASE
                / "the progress coordinate_gradient.py",
            },
            "two_state_generation_shared": dict(_TWO_STATE_GENERATION_SHARED),
            # ``outcome`` defines the reported claim and the pre-registered design gate,
            # ``disjoint_rmsd`` the motif-disjoint fidelity measurement K-15 requires beside it,
            # and ``validation`` the confusion matrices and balanced accuracy the backbone-only
            # classifier's claim rests on. None is read by the training loop, so a change to any
            # of them re-keys an evaluation and retires no checkpoint -- which is why an
            # evaluation group can be declared generously at no cost.
            "kinase_evaluation_only": {
                "kinase_disjoint_rmsd": _KINASE / "disjoint_rmsd.py",
                "kinase_outcome": _KINASE / "outcome.py",
                "kinase_validation": _KINASE / "validation.py",
            },
        },
        # The kinase composition names the cryptic GENERATION-SHARED group, and that is not an
        # oversight: a kinase run executes ``cryptic_steering/pocket.py`` (the pair geometry the progress coordinate
        # is built from) and ``cryptic_steering/objective.py`` (``validate_the progress coordinate_curvature``, the
        # D-31 bound the off-path weight is checked against), so leaving them out would be
        # fail-OPEN -- a change to the pair builder would move no kinase digest while changing
        # every kinase coordinate. It binds a kinase checkpoint to four cryptic files a kinase
        # run does not execute as well, and that over-binding is accepted in the fail-closed
        # direction rather than split into a sixth group.
        #
        # It does NOT name ``cryptic_training_only``: a kinase arm constructs
        # ``KinaseSteeringLoss`` and never imports the cryptic loss.
        code_group_composition={
            "training": (
                "training_only",
                "generation_shared",
                "two_state_generation_shared",
                "cryptic_generation_shared",
                "kinase_training_only",
                "kinase_generation_shared",
            ),
            "evaluation": ("evaluation_only", "kinase_evaluation_only"),
        },
        # ``perp_loss`` and ``dihedral_loss`` have been written since those terms existed.
        # ``margin_loss`` is NEW and closes a hole this pass is the first opportunity to close:
        # ``KinaseSteeringLossComponents.margin`` reached the in-memory components, the objective
        # identity and a log WARNING, and no per-row column at all, so a reader of the journal
        # could not tell whether the classifier-region term had ever fired. It is gated on
        # ``lambda_margin``, so an arm that does not price it writes the byte-identical row.
        journal_columns={
            "perp_loss": "lambda_perp",
            "dihedral_loss": "lambda_dihedral",
            "margin_loss": "lambda_margin",
        },
        # The ONE non-default spelling this property's selection hook can return, admitted here so
        # that the completion gate -- which holds a persisted ``adapter_config`` and no objective
        # -- can validate a checkpoint that recorded it. Every kinase arm shipped before
        # The current arms run at ``lambda_margin = 0`` and select on the shared ``target_mae_sigma``,
        # so this widens what is expressible and changes no arm on disk.
        #
        # It is ``H``, the squared violation of the eight affine margins whose simultaneous
        # nonnegativity IS the Kincore label, meaned over the round's held-out rows. A margin arm
        # may train at ``lambda_the progress coordinate = 0``; under the shared default it would then choose its shipped
        # epoch on the very ``the progress coordinate`` proxy it was built to stop depending on, which K-22 measured
        # moving OPPOSITE to that arm's own objective over 40 epochs.
        selection_metric_names=("margin_hinge_mean",),
    ),
    "domain_motion_backbone_rmsd": PropertyRegistration(
        name="domain_motion_backbone_rmsd",
        descriptor_module="sae_bioemu.domain_motion.property",
        descriptor_attribute="DOMAIN_MOTION_PROPERTY",
        arm_config_key="domain_motion_arm",
        code_group_members={
            # ONE group, and that is a statement about this property rather than an economy.
            # Every file below shapes the trained adapter and is read by no scorer: domain motion
            # has no evaluator of its own yet (``bioemu-benchmarks``' success@B over N/CA/C/O is
            # the metric, and the registry refuses ``evaluate`` by name until a scorer exists), so
            # a ``domain_motion_evaluation_only`` group would be a declaration with nothing in it.
            # Training-only rather than generation-shared for the reason ``cryptic_training_only``
            # is: no evaluation computes a loss, so a change here cannot move a generated structure
            # or a scored number and must not be able to invalidate an evaluation.
            #
            #   artifact    holds the two deposited conformers, the metric-set mask and each
            #               system's s_p, so it decides WHICH ATOMS the metric is read over.
            #   frames      reconstructs N/CA/C/O from the denoiser's per-residue frames, so it
            #               decides what the measured structure IS.
            #   loss        the objective class, and the contract it holds the shared loss to.
            #   objective   DEFINES the metric, the huberised hinge, tau_p and which conformer a
            #               level names.
            #   property    this property's descriptor: the dose set, the target origin, the
            #               callback versions, the objective class and the refused weights.
            #   separation  the pool gate: which systems are trainable at all.
            "domain_motion_training_only": {
                "domain_motion_artifact": _DOMAIN_MOTION / "artifact.py",
                "domain_motion_frames": _DOMAIN_MOTION / "frames.py",
                "domain_motion_loss": _DOMAIN_MOTION / "loss.py",
                "domain_motion_objective": _DOMAIN_MOTION / "objective.py",
                "domain_motion_property": _DOMAIN_MOTION / "property.py",
                "domain_motion_separation": _DOMAIN_MOTION / "separation.py",
            },
            # BYTE-IDENTICAL to the cryptic declaration, deliberately. ``scientific_code_paths``
            # merges the two and refuses two declarations of one member as different files, so the
            # sharing is enforced rather than agreed; that refusal is the reason the duplication is
            # safe to write twice instead of routing both through a shared file that would then
            # need its own group.
            "aligned_rmsd_shared": dict(_ALIGNED_RMSD_SHARED),
            # THE HELD-OUT SCORER (D-89). ``success_at_b.py`` is ``bioemu-benchmarks``' own
            # success@B over N/CA/C/O on the GLOBAL path at tau = 3.0: it decides which atoms the
            # reported number is measured over, in which residue frame, under which superposition,
            # and how a per-system rate is aggregated.
            #
            # The comment on ``domain_motion_training_only`` above said this group "would be a
            # declaration with nothing in it" because domain motion had no evaluator. It has one
            # now, and the group exists for exactly the reason the cryptic and kinase evaluation
            # groups do: a change to a scorer must be able to re-key an EVALUATION and must never
            # retire a trained adapter. Nothing here is read by the training loop.
            #
            # ``benchmark.py`` is deliberately NOT here even though it reads the same assets. It is
            # imported by ``domain_motion/__init__.py`` and therefore executed by every domain-
            # motion TRAINING run, so hashing it as evaluation-only would be fail-open; it also
            # keeps only C-alpha and C-beta, which is why the scorer does its own four-atom read.
            #
            # ``gap_closed.py`` is the PAIRED scorer: the fraction of each system's own gap the
            # intervention closed against a matched unsteered control at the same ``draw_seed``,
            # with success@tau demoted to a secondary over a baseline-admitted population and the
            # reachability bound reported as a stratum. D-94 measured the fixed threshold on its
            # own six-system panel and found three systems inside tau with no adapter at all and
            # one unreachable at any dose, so it decides what the reported NUMBER is quite as much
            # as ``success_at_b.py`` does and belongs in the same group for the same reason.
            #
            # It measures nothing itself: the distance comes from
            # ``domain_motion/objective.aligned_backbone_rmsd`` over ``steering/aligned_rmsd.py``,
            # both of which are in this property's TRAINING composition. So a change to what the
            # metric MEANS re-keys the checkpoint the paired report is about, and the report is
            # invalidated through that identity rather than through a second copy of the kernel.
            "domain_motion_evaluation_only": {
                "domain_motion_gap_closed": _DOMAIN_MOTION / "gap_closed.py",
                "domain_motion_success_at_b": _DOMAIN_MOTION / "success_at_b.py",
            },
        },
        # It does NOT name ``two_state_generation_shared``, and the omission is the content of the
        # entry. That group is ``bioemu/cryptic_generation.py``, which reconstructs the atoms the
        # two PROGRESS-COORDINATE properties measure between; a domain-motion run executes not one
        # line of it, because the direct metric reads N/CA/C/O through ``domain_motion/frames.py``
        # instead. Naming it would bind every domain-motion checkpoint to a file it never runs,
        # which is the over-binding half of the defect the registry exists to end.
        #
        # It does not name ``cryptic_generation_shared`` either: there is no pair set, no
        # transition vector and no curvature bound on this path.
        code_group_composition={
            "training": (
                "training_only",
                "generation_shared",
                "aligned_rmsd_shared",
                "domain_motion_training_only",
            ),
            # The EVALUATION composition gains this property's own scorer group and NOTHING ELSE.
            # It shares no group with the training composition above, so adding it moves this
            # property's EVALUATION fingerprint and cannot move any training fingerprint -- its
            # own or another property's. ``tests/test_domain_motion_success_at_b.py`` proves that
            # by group digest rather than by inspection.
            "evaluation": ("evaluation_only", "domain_motion_evaluation_only"),
        },
        # ONE column, carrying the UNCLAMPED metric in Angstrom. The hinge is flat below
        # ``tau_p``, so a hinge of zero says nothing about whether the draw sits at 2.9 Angstrom
        # or at 0.4, and an arm whose only recorded term is the hinge is undiagnosable exactly
        # where it is succeeding.
        #
        # It is gated on ``lambda_m`` (D-87): THE SAME NAME AND THE SAME MEANING the cryptic pocket
        # prices its graded metric under -- the multiplier on that property's own
        # ``metric_mse``/``huberised_hinge``. It used to be gated on ``direct_metric_weight``, a
        # restatement of the shared ``lambda_y``, because a shared weight cannot gate a column:
        # every objective there has ever been carries a positive ``lambda_y``, so the column would
        # be demanded of every property's rows including those that cannot compute it. That
        # restatement was the third spelling of one number and it is gone; ``lambda_y`` is now zero
        # on this path and refused if it is not, because this objective has no progress-coordinate
        # target for it to price.
        #
        # A distinct column NAME from the cryptic ``metric_deviation_angstrom``, even though both
        # are a superposed backbone deviation in Angstrom and both travel on the same sample
        # field. One name may carry one gate -- ``journal_column_weights`` refuses a column
        # declared against two weights -- and these two are gated by different ones. They are also
        # not the same quantity: the cryptic one is measured over a pocket's metric region against
        # a deposited cryptic conformer and this one over a domain-motion metric set against a
        # deposited conformer pair, so a single column would pool two measurements under one name.
        journal_columns={"direct_metric_angstrom": "lambda_m"},
        # ``lambda_m``, CO-OWNED with the cryptic pocket. ``optional_runtime_key_owners`` admits
        # more than one owner for one key exactly when the two price the SAME quantity under the
        # same name, which is the claim D-87 makes: on both properties ``lambda_m`` multiplies that
        # property's own graded Angstrom metric, per observation, meaned over the batch.
        #
        # ``metric_tau_angstrom`` is deliberately NOT claimed here. Domain motion's threshold lives
        # in the ARM artifact and reaches no part of the default MSE form, and tau is an EVALUATION
        # parameter on this path (D-85).
        optional_runtime_keys=("lambda_m",),
        # This property selects on its METRIC and never on the target error, unconditionally, and
        # the reason is a property of the objective rather than of a weight: the target error here
        # IS the hinge, which is identically zero over the whole region a successful arm lives in.
        # So the vocabulary has one entry and the descriptor's hook always returns it.
        selection_metric_names=("direct_metric_angstrom_mean",),
    ),
    "kinase_conformer_rmsd": PropertyRegistration(
        name="kinase_conformer_rmsd",
        descriptor_module="sae_bioemu.kinase_rmsd.property",
        descriptor_attribute="KINASE_RMSD_PROPERTY",
        # ITS OWN KEY. ``arm_config_keys`` builds a plain dict, so a duplicate of ``kinase_arm``,
        # ``cryptic_arm`` or ``domain_motion_arm`` would SILENTLY OVERWRITE rather than raise, and
        # one of the two properties would then own a key it does not answer for.
        arm_config_key="kinase_rmsd_arm",
        code_group_members={
            # ONE group, and it is a statement about this property rather than an economy. Every
            # file below shapes the trained adapter and is read by no scorer: this property has no
            # evaluator of its own (the registry refuses ``evaluate`` by name until one exists), so
            # a ``kinase_rmsd_evaluation_only`` group would be a declaration with nothing in it.
            # Training-only rather than generation-shared for the reason ``cryptic_training_only``
            # is: no evaluation computes a loss, so a change here cannot move a generated structure
            # or a scored number and must not be able to invalidate an evaluation.
            #
            #   artifact    holds the two deposited chains PRE-RESTRICTED to the metric set, the
            #               metric indices and each kinase's s_p, so it decides WHICH ATOMS the
            #               metric is read over. It also declares ``KinaseRmsdDefinition``, the
            #               type the shared generation path is handed.
            #   frames      reconstructs N/CA/C/O from the denoiser's per-residue frames, so it
            #               decides what the measured structure IS.
            #   loss        the objective class, and the contract it holds the shared loss to.
            #   objective   DEFINES the metric, the two forms, tau, the pool gate, the three
            #               candidate index sets and which deposited chain a level names.
            #   property    this property's descriptor: the dose set, the target origin, the
            #               callback versions, the objective class and the refused weights.
            #
            # ``__init__`` is NOT a member, and the omission is deliberate: it re-exports names and
            # decides nothing, so hashing it would retire checkpoints for an edit to a docstring.
            "kinase_rmsd_training_only": {
                "kinase_rmsd_artifact": _KINASE_RMSD / "artifact.py",
                "kinase_rmsd_frames": _KINASE_RMSD / "frames.py",
                "kinase_rmsd_loss": _KINASE_RMSD / "loss.py",
                "kinase_rmsd_objective": _KINASE_RMSD / "objective.py",
                "kinase_rmsd_property": _KINASE_RMSD / "property.py",
            },
            # BYTE-IDENTICAL to the cryptic and domain-motion declarations, deliberately.
            # ``scientific_code_paths`` merges the three and refuses two declarations of one member
            # as different files, so the sharing is enforced rather than agreed -- and because the
            # merge adds no member to an existing group, naming it here moves NO existing
            # property's digest. That is the whole basis for this registration being free.
            "aligned_rmsd_shared": dict(_ALIGNED_RMSD_SHARED),
        },
        # It names NO ``kinase_*`` group and NO ``cryptic_*`` group, and that omission is the
        # content of the entry. ``kinase_generation_shared`` is the DFG PROGRESS COORDINATE's
        # machinery -- the pair geometry, the motif window, the Kincore centroid table, the torus
        # embedding -- and this property executes not one line of it: it never imports
        # ``outcome.py``, ``kincore.py`` or the BBAminus classifier, because its quantity is an
        # Angstrom distance to a deposited chain rather than a label or a coordinate. Naming it
        # would bind every kinase-RMSD checkpoint to files it never runs, which is the
        # over-binding half of the defect the registry exists to end -- and would be read as a
        # claim that the two kinase properties measure the same thing.
        #
        # It does not name ``two_state_generation_shared`` either. That group is
        # ``bioemu/cryptic_generation.py``, which reconstructs the atoms the two PROGRESS-
        # COORDINATE properties measure between; this property reads N/CA/C/O through
        # ``kinase_rmsd/frames.py`` instead, which is why that file is a copy rather than an
        # import.
        code_group_composition={
            "training": (
                "training_only",
                "generation_shared",
                "aligned_rmsd_shared",
                "kinase_rmsd_training_only",
            ),
            "evaluation": ("evaluation_only",),
        },
        # ONE column, carrying the UNCLAMPED metric in Angstrom, gated on ``lambda_m``.
        #
        # A DISTINCT NAME from the cryptic ``metric_deviation_angstrom`` and the domain-motion
        # ``direct_metric_angstrom``, even though all three are a superposed backbone deviation in
        # Angstrom travelling on the same sample field. One name may carry one gate --
        # ``journal_column_weights`` refuses a column declared against two weights -- and beyond
        # that they are not the same quantity: the cryptic one is measured over a pocket's metric
        # region against a deposited cryptic conformer, the domain-motion one over a DynDom metric
        # set, and this one over a kinase ACTIVATION LOOP against a Kincore-labelled chain. A
        # single column would pool three measurements under one name.
        journal_columns={"kinase_conformer_rmsd_angstrom": "lambda_m"},
        # ``lambda_m``, CO-OWNED with the cryptic pocket and domain motion.
        # ``optional_runtime_key_owners`` admits more than one owner for one key exactly when they
        # price the SAME quantity under the same name, which is the case here: on all three
        # properties ``lambda_m`` multiplies that property's own graded Angstrom metric, per
        # observation, meaned over the batch.
        #
        # ``metric_tau_angstrom`` is deliberately NOT claimed. This property's threshold is a
        # constant in its own objective module and reaches no part of the default MSE form; tau is
        # an EVALUATION parameter on this path.
        optional_runtime_keys=("lambda_m",),
        # This property selects on its METRIC and never on the target error, unconditionally.
        # Under the hinge the target error IS the hinge, identically zero over the whole region a
        # successful arm lives in; under the MSE it is the metric divided by a per-protein
        # constant. And K-22 measured the other kinase property's default ``target_mae_sigma``
        # selection improving for 40 epochs while the graded quantity moved the other way.
        selection_metric_names=("direct_metric_angstrom_mean",),
    ),
    "pocket_conformer_rmsd": PropertyRegistration(
        name="pocket_conformer_rmsd",
        descriptor_module="sae_bioemu.cryptic_rmsd.property",
        descriptor_attribute="POCKET_RMSD_PROPERTY",
        # ITS OWN KEY, and the collision that matters most is ``cryptic_arm``: both properties are
        # about cryptic pockets, and the two loaders open entirely different files -- that one
        # reads a pocket pair set, a transition vector, an apo sidecar and a site source, this one
        # reads a table of deposited N/CA/C/O conformers. ``arm_config_keys`` is a plain dict
        # comprehension, so a duplicate would SILENTLY OVERWRITE rather than raise.
        arm_config_key="pocket_rmsd_arm",
        code_group_members={
            # ONE group, and it is a statement about this property rather than an economy. Every
            # file below shapes the trained adapter and is read by no scorer: this property has no
            # evaluator of its own (the registry refuses ``evaluate`` by name until one exists), so
            # a ``cryptic_rmsd_evaluation_only`` group would be a declaration with nothing in it.
            # Training-only rather than generation-shared for the reason ``cryptic_training_only``
            # is: no evaluation computes a loss, so a change here cannot move a generated structure
            # or a scored number and must not be able to invalidate an evaluation.
            #
            #   artifact    holds the two deposited conformers PRE-RESTRICTED to the metric region,
            #               the metric indices, each protein's sequence and its s_p, so it decides
            #               WHICH ATOMS the deviation is read over. It also declares
            #               ``PocketRmsdDefinition``, the type the shared generation path is handed.
            #   frames      reconstructs N/CA/C/O from the denoiser's per-residue frames, so it
            #               decides what the measured structure IS.
            #   loss        the objective class, and the contract it holds the shared loss to.
            #   objective   DEFINES the deviation, the per-system normalisation, the pool gate and
            #               which deposited conformer a level names.
            #   property    this property's descriptor: the dose set, the TARGET ORIGIN, the
            #               callback versions, the objective class and the refused weights.
            #
            # ``__init__`` is NOT a member, and the omission is deliberate: it re-exports names and
            # decides nothing, so hashing it would retire checkpoints for an edit to a docstring.
            "cryptic_rmsd_training_only": {
                "cryptic_rmsd_artifact": _CRYPTIC_RMSD / "artifact.py",
                "cryptic_rmsd_frames": _CRYPTIC_RMSD / "frames.py",
                "cryptic_rmsd_loss": _CRYPTIC_RMSD / "loss.py",
                "cryptic_rmsd_objective": _CRYPTIC_RMSD / "objective.py",
                "cryptic_rmsd_property": _CRYPTIC_RMSD / "property.py",
            },
            # BYTE-IDENTICAL to the cryptic, domain-motion and kinase-RMSD declarations,
            # deliberately. ``scientific_code_paths`` merges the four and refuses two declarations
            # of one member as different files, so the sharing is enforced rather than agreed --
            # and because the merge adds no member to an existing group, naming it here moves NO
            # existing property's digest. That is the whole basis for this registration being free.
            "aligned_rmsd_shared": dict(_ALIGNED_RMSD_SHARED),
        },
        # It names NO ``cryptic_*`` group, and that omission is the content of the entry. A run of
        # this property executes not one line of ``cryptic_steering``: no pair set, no transition
        # vector, no D-31 curvature bound, no apo sidecar, no stage-1 setpoint table, no site-field
        # cache and no ``CrypticSteeringLoss``. Naming ``cryptic_generation_shared`` or
        # ``cryptic_training_only`` would bind every ``pocket_conformer_rmsd`` checkpoint to files
        # it never runs, which is the over-binding half of the defect the registry exists to end --
        # and would be read as a claim that the two cryptic properties measure the same thing.
        #
        # It does not name ``two_state_generation_shared`` either. That group is
        # ``bioemu/cryptic_generation.py``, which reconstructs the atoms the two PROGRESS-
        # COORDINATE properties measure between; this property reads N/CA/C/O through
        # ``cryptic_rmsd/frames.py`` instead, which is why that file is a copy rather than an
        # import.
        code_group_composition={
            "training": (
                "training_only",
                "generation_shared",
                "aligned_rmsd_shared",
                "cryptic_rmsd_training_only",
            ),
            "evaluation": ("evaluation_only",),
        },
        # ONE column, carrying the UNCLAMPED deviation in Angstrom, gated on ``lambda_m``.
        #
        # A DISTINCT NAME from the cryptic ``metric_deviation_angstrom``, the domain-motion
        # ``direct_metric_angstrom`` and the kinase ``kinase_conformer_rmsd_angstrom``, even though
        # all four are a superposed backbone deviation in Angstrom travelling on the same sample
        # field. One name may carry one gate -- ``journal_column_weights`` refuses a column declared
        # against two weights -- and beyond that they are not the same quantity. The closest of the
        # four is the cryptic one, measured over the same metric region against the same deposited
        # conformers; it is still a different measurement, because it is journalled by an arm whose
        # TARGETS came from a ``the progress coordinate`` state against a matched control while this one's came from
        # ``s_p``, so pooling them would compare two objectives under one column.
        journal_columns={"pocket_conformer_rmsd_angstrom": "lambda_m"},
        # ``lambda_m``, CO-OWNED with the cryptic pocket, domain motion and kinase RMSD.
        # ``optional_runtime_key_owners`` admits more than one owner for one key exactly when they
        # price the SAME quantity under the same name, which is the case here: on all four
        # properties ``lambda_m`` multiplies that property's own graded Angstrom metric, per
        # observation, meaned over the batch, with the MSE convention factor kept out of the weight
        # and in the arithmetic.
        #
        # ``metric_tau_angstrom`` is deliberately NOT claimed, and on this property the omission is
        # sharper than elsewhere: there is no hinge form at all. tau has two values of record and no
        # default on this pool (D-63 measured 30.5/60.6 at tau = 1.0 against 49/85 at tau = 1.5 on
        # one model and one set of 33 systems), so a threshold in the objective would make the two
        # readings of one run two readings of two differently-trained adapters -- and under the
        # system_constant origin it would enter the per-system target table's digest as well.
        optional_runtime_keys=("lambda_m",),
        # This property selects on its DEVIATION and never on the target error, unconditionally.
        # Here the target error IS the deviation divided by a per-protein constant, so selecting on
        # it ranks epochs by a normalised quantity rather than by the Angstrom one a report quotes.
        # Its own spelling rather than the ``direct_metric_angstrom_mean`` domain motion and kinase
        # RMSD share: a selection metric name is a persisted identity key.
        selection_metric_names=("pocket_conformer_rmsd_angstrom_mean",),
    ),
}
