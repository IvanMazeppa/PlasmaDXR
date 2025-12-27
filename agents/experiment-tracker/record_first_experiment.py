#!/usr/bin/env python3
"""
Record the domain_scale experiment we just ran.

This populates the knowledge base with our first learned relationship:
- Increasing domain_scale without adjusting position shifts content upward
"""

from tracker import ExperimentTracker

def main():
    tracker = ExperimentTracker()

    # Start session
    session_id = tracker.start_session(
        asset_name="dramatic_explosion_e2e",
        effect_type="explosion",
        description="A powerful fiery explosion with bright orange-red flames",
        reference_path="/home/maz3ppa/projects/PlasmaDXR/assets/reference_images/explosions/explosion_reference_web_1.jpg",
        semantic_query="a dramatic fiery explosion with intense orange flames, billowing black smoke"
    )
    print(f"Started session: {session_id}")

    # Record baseline (v1)
    tracker.record_baseline(
        params={
            'domain_scale': 6.0,
            'domain_location_z': 0,
            'camera_location': [6, -6, 4],
            'resolution': 96,
            'flame_smoke': 1.0,
            'flame_max_temp': 5.0
        },
        scores={
            'clip': 0.6459,
            'aesthetic': 6.01,
            'gradient_signal': 0.6565,
            'lpips': 0.8292,
            'temporal_consistency': 0.877
        },
        render_path='/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/dramatic_explosion_e2e/render_0035.png',
        script_path='/home/maz3ppa/projects/PlasmaDXR/assets/blender_scripts/generated/dramatic_explosion_e2e.py'
    )
    print("Recorded baseline (v1)")

    # Record experiment result (v2)
    result = tracker.record_experiment(
        hypothesis="Increasing domain_scale will fix clipping at top edge",
        issue_addressed="Smoke/flames being clipped at top edge of frame",
        result_params={
            'domain_scale': 10.0,  # Increased from 6.0
            'domain_location_z': 2,  # Offset upward (this was the mistake!)
            'camera_location': [8, -8, 5],  # Pulled back
            'resolution': 96,
            'flame_smoke': 1.0,
            'flame_max_temp': 5.0
        },
        result_scores={
            'clip': 0.6368,
            'aesthetic': 5.96,
            'gradient_signal': 0.6445,
            'lpips': 0.8378,
            'temporal_consistency': 0.886
        },
        result_render='/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/dramatic_explosion_e2e_v2/render_0035.png',
        result_script='/home/maz3ppa/projects/PlasmaDXR/assets/blender_scripts/generated/dramatic_explosion_e2e_v2.py',
        success=False,  # Goal not achieved - framing got worse
        observed_effects=[
            {
                'metric': 'framing_position',
                'direction': 'shifted_up',
                'magnitude': 0.33,  # Content now in top 2/3 of frame
                'expected': False,
                'side_effect': True,
                'description': 'Content shifted into top 2/3 of frame instead of being centered'
            },
            {
                'metric': 'top_clipping',
                'direction': 'reduced',
                'magnitude': 0.5,
                'expected': True,
                'side_effect': False,
                'description': 'Clipping at top edge was reduced'
            },
            {
                'metric': 'smoke_visibility',
                'direction': 'decreased',
                'magnitude': 0.2,
                'expected': False,
                'side_effect': True,
                'description': 'Explosion appears to lack smoke compared to v1'
            }
        ],
        learnings=[
            "Increasing domain_scale alone shifts content position - need to compensate",
            "Domain location offset (z=2) pushes content upward in frame",
            "When fixing clipping, adjust domain height WITHOUT changing domain center position",
            "Formula: Keep domain_location_z = 0, just increase domain height via scale"
        ],
        warnings=[
            "WARNING: domain_scale increase + domain_location_z offset = content shifts up",
            "WARNING: Camera pullback (6,6,4 -> 8,8,5) was not enough to compensate",
            "Do NOT offset domain_location when just trying to fix clipping"
        ],
        human_notes="User noticed framing got worse, not better. Content now in top 2/3 of frame."
    )

    print(f"\nExperiment recorded: {result.experiment_id}")
    print(f"Success: {result.success}")
    print(f"Partial success: {result.partial_success}")
    print(f"\nScore changes:")
    for metric, change in result.score_changes.items():
        print(f"  {metric}: {change:+.4f}")

    print(f"\nLearnings recorded: {len(result.learnings)}")
    print(f"Warnings recorded: {len(result.warnings)}")

    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")

    # Now let's add a manual learning about the correct approach
    tracker.add_manual_learning(
        parameter='domain_scale',
        rule="To fix clipping without shifting content: increase domain scale but keep domain_location_z at 0. The domain expands symmetrically from center.",
        warning="Changing domain_location_z will shift where content appears in the frame",
        context="Fixing clipping issues in pyro/explosion simulations"
    )
    print("\nAdded manual learning about correct domain_scale approach")

    # End session
    tracker.end_session(
        final_status="learning_recorded",
        best_score=0.6459  # v1 CLIP score was actually better
    )

    # Show final statistics
    print("\n" + "="*60)
    print("KNOWLEDGE BASE STATUS")
    print("="*60)

    stats = tracker.get_statistics()
    print(f"\nTotal experiments: {stats['total_experiments']}")
    print(f"Parameters tracked: {stats['parameters_tracked']}")
    print(f"Causal relationships: {stats['causal_relationships']}")

    # Query the knowledge base
    print("\n" + "-"*60)
    print("Knowledge about 'domain_scale':")
    print("-"*60)
    info = tracker.get_parameter_info('domain_scale')
    print(f"  Experiments: {info.get('experiments_count', 0)}")
    print(f"  Success rate: {info.get('success_rate', 'N/A')}")
    print(f"  Rules:")
    for rule in info.get('rules', []):
        print(f"    - {rule}")
    print(f"  Warnings:")
    for warning in info.get('warnings', []):
        print(f"    - {warning}")

    # Test suggestion system
    print("\n" + "-"*60)
    print("Testing suggestion system with 'clipping' issue:")
    print("-"*60)
    suggestions = tracker.suggest_experiments(
        issue="clipping at top edge",
        current_params={'domain_scale': 6.0},
        current_scores={'clip': 0.64}
    )
    for s in suggestions:
        print(f"\n  Hypothesis: {s.hypothesis}")
        print(f"  Confidence: {s.confidence:.0%}")
        if s.risks:
            print(f"  RISKS:")
            for risk in s.risks:
                print(f"    - {risk}")


if __name__ == "__main__":
    main()
