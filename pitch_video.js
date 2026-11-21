const plot = document.getElementById('{plot_id}');
const gameFeeds = new Map();
const feedBase = 'https://baseballsavant.mlb.com/gf?game_pk=';
const videoBase = 'https://baseballsavant.mlb.com/sporty-videos?playId=';
const uuidPattern = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i;

async function gamePitches(gameId) {
    if (!gameFeeds.has(gameId)) {
        const request = fetch(feedBase + gameId, {
            credentials: 'omit',
            signal: AbortSignal.timeout(12000),
        }).then(response => {
            if (!response.ok) throw new Error('Savant game feed unavailable');
            return response.json();
        }).then(game => [
            ...Object.values(game.home_pitchers).flat(),
            ...Object.values(game.away_pitchers).flat(),
        ]).catch(error => {
            gameFeeds.delete(gameId);
            throw error;
        });
        gameFeeds.set(gameId, request);
    }
    return gameFeeds.get(gameId);
}

plot.on('plotly_click', async ({points}) => {
    const key = points.find(point => point.data.type === 'scattergl')?.customdata;
    if (!Array.isArray(key) || key.length !== 5 || !key.every(Number.isInteger)) return;
    const [gameId, atBat, number, catcher, batter] = key;

    // Open during the user gesture so the asynchronous Savant lookup can navigate it.
    const tab = window.open('about:blank', '_blank');
    if (!tab) {
        window.alert('Allow pop-ups for Catcher Lab to open pitch videos.');
        return;
    }
    tab.opener = null;
    tab.document.title = 'Opening Baseball Savant';
    tab.document.body.textContent = 'Finding the pitch on Baseball Savant…';
    try {
        const pitches = await gamePitches(gameId);
        const ids = new Set(pitches.filter(pitch =>
            Number(pitch.ab_number) === atBat && Number(pitch.pitch_number) === number &&
            Number(pitch.catcher) === catcher && Number(pitch.batter) === batter &&
            uuidPattern.test(pitch.play_id)
        ).map(pitch => pitch.play_id.toLowerCase()));
        if (ids.size === 1) {
            tab.location.replace(videoBase + ids.values().next().value);
        } else {
            tab.document.body.textContent = 'Baseball Savant has no matching clip for this pitch.';
        }
    } catch {
        tab.document.body.textContent = 'Baseball Savant is temporarily unavailable. Close this tab and try the pitch again.';
    }
});
