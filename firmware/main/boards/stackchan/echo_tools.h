#ifndef _STACKCHAN_ECHO_TOOLS_H
#define _STACKCHAN_ECHO_TOOLS_H

class CoreS3AudioCodec;
class McpServer;

/**
 * MCP tools for switching echo cancellation and for measuring what it does:
 *
 *   self.audio.set_echo_cancellation   put the canceller and its software
 *                                      reference in or out of the voice path
 *   self.audio.get_echo_stats          state of the reference, and levels
 *                                      before and after the voice front end
 *   self.audio.echo_probe              record a moment of playback and report
 *                                      how the microphone relates to the
 *                                      reference
 *   self.audio.set_echo_options        store how the front end is built
 *                                      (canceller mode, cores), for the next
 *                                      boot
 *
 * The measuring ones exist because the first attempt at echo cancellation on
 * this board was judged by holding a conversation with it, which says "the
 * microphone is dead" and nothing about why.
 */
void RegisterEchoTools(McpServer& mcp_server, CoreS3AudioCodec* codec);

#endif  // _STACKCHAN_ECHO_TOOLS_H
