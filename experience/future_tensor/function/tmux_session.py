"""
tmux_session :=
    $tmux_session_prefix str
    # inline

Global configuration for tmux-based FutureTensor ops.
"""

# Default prefix for tmux session names.
# Sessions are named f"{tmux_session_prefix}{instance_id}".
tmux_session_prefix: str = "ft_tmux_"
