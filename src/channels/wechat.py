"""WeChat channel implementation with parallel support via spawn."""
import asyncio
import hashlib
import time
import base64
import xml.etree.ElementTree as ET
from typing import Any, Optional, Dict, List
from urllib.parse import parse_qs
import aiohttp
from Crypto.Cipher import AES
from .base import Channel, ChannelType, InboundMessage, OutboundMessage


class WeChatChannel(Channel):
    """WeChat bot channel with access token management."""

    def __init__(self, config: dict, spawn_tool=None):
        """Initialize WeChat channel.

        Args:
            config: Configuration with 'corp_id', 'corp_secret', 'agent_id',
                    'token', 'encoding_aes_key'
            spawn_tool: Optional SpawnTool instance for parallel message processing
        """
        super().__init__(ChannelType.WECHAT, config)
        self.corp_id = config.get("corp_id")
        self.corp_secret = config.get("corp_secret")
        self.agent_id = config.get("agent_id")
        self.token = config.get("token")
        self.encoding_aes_key = config.get("encoding_aes_key")
        self.enable_parallel = config.get("enable_parallel", True)
        self._access_token: Optional[str] = None
        self._token_expires_at = 0
        self._session: Optional[aiohttp.ClientSession] = None
        self._spawn_tool = spawn_tool
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start WeChat bot."""
        if not self.corp_id or not self.corp_secret or not self.agent_id:
            raise ValueError("WeChat corp_id, corp_secret, and agent_id are required")

        self._session = aiohttp.ClientSession()
        await self._get_access_token()
        self._running = True
        
        # Start polling for messages
        self._poll_task = asyncio.create_task(self._poll_messages())
        print(f"WeChat channel started (corp_id: ...{self.corp_id[-4:]})")

    async def stop(self) -> None:
        """Stop WeChat channel."""
        self._running = False
        self._access_token = None
        
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        
        if self._session:
            await self._session.close()
            self._session = None
        
        print("WeChat channel stopped")

    async def send_message(self, message: OutboundMessage) -> str:
        """Send message to WeChat.
        
        Args:
            message: OutboundMessage with chat_id as user_id or department_id
            
        Returns:
            Message ID
        """
        payload = {
            "touser": message.chat_id,
            "msgtype": "text",
            "agentid": self.agent_id,
            "text": {"content": message.content},
            "safe": 0,
        }

        result = await self._api_request("message/send", method="POST", payload=payload)
        return result.get("msgid", "")

    async def send_batch_messages(self, messages: List[OutboundMessage]) -> Dict[str, str]:
        """Send multiple messages, optionally in parallel.
        
        Args:
            messages: List of OutboundMessage objects
            
        Returns:
            Dict mapping chat_id to message_id
        """
        if self.enable_parallel and self._spawn_tool and len(messages) > 1:
            # Use spawn to process messages in parallel
            task_description = f"Send {len(messages)} WeChat messages in parallel"
            await self._spawn_tool.execute(
                task=f"Process {len(messages)} WeChat messages with IDs: {[m.chat_id for m in messages]}",
                label=task_description
            )
        
        # Send messages concurrently
        results = await asyncio.gather(
            *[self.send_message(msg) for msg in messages],
            return_exceptions=True
        )
        
        return {
            msg.chat_id: result 
            for msg, result in zip(messages, results)
            if not isinstance(result, Exception)
        }

    async def verify_signature(self, msg_signature: str, timestamp: str, nonce: str) -> bool:
        """Verify WeChat webhook signature.
        
        Args:
            msg_signature: Signature from WeChat
            timestamp: Timestamp from WeChat
            nonce: Nonce from WeChat
            
        Returns:
            True if signature is valid
        """
        if not self.token:
            return False

        # Sort and concatenate token, timestamp, nonce
        data = sorted([self.token, timestamp, nonce])
        string_to_sign = "".join(data)
        
        # SHA1 hash
        signature = hashlib.sha1(string_to_sign.encode()).hexdigest()
        
        return signature == msg_signature

    async def _poll_messages(self) -> None:
        """Poll WeChat for messages (polling mode)."""
        # WeChat callback typically uses webhook, but we can implement
        # periodic checking of message history for certain scenarios
        while self._running:
            try:
                await asyncio.sleep(30)  # Poll every 30 seconds
                # In production, you might check specific endpoints
                # based on WeChat's message callback configuration
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error polling WeChat: {e}")
                await asyncio.sleep(5)

    def _decrypt_message(self, msg_encrypt: str) -> str:
        """Decrypt WeChat message using AES-CBC.
        
        Args:
            msg_encrypt: Base64 encoded encrypted message
            
        Returns:
            Decrypted XML message
            
        Raises:
            ValueError: If decryption fails
        """
        if not self.encoding_aes_key:
            raise ValueError("encoding_aes_key is required for message decryption")

        # Decode base64
        try:
            encrypted_bytes = base64.b64decode(msg_encrypt)
        except Exception as e:
            raise ValueError(f"Failed to decode base64: {e}")

        # AES key is base64 encoded in config, need to decode it
        try:
            aes_key = base64.b64decode(self.encoding_aes_key + "=")
        except Exception as e:
            raise ValueError(f"Failed to decode AES key: {e}")

        if len(aes_key) != 32:
            raise ValueError(f"AES key length must be 32, got {len(aes_key)}")

        # Extract IV (first 16 bytes)
        iv = encrypted_bytes[:16]
        ciphertext = encrypted_bytes[16:]

        try:
            # Decrypt using AES-CBC
            cipher = AES.new(aes_key, AES.MODE_CBC, iv)
            decrypted = cipher.decrypt(ciphertext)
            
            # Remove PKCS7 padding
            padding_length = decrypted[-1]
            if isinstance(padding_length, str):
                padding_length = ord(padding_length)
            
            message = decrypted[:-padding_length].decode('utf-8')
            return message
        except Exception as e:
            raise ValueError(f"Failed to decrypt message: {e}")

    def _parse_xml_message(self, xml_content: str) -> Dict[str, str]:
        """Parse WeChat XML message.
        
        Args:
            xml_content: XML message content
            
        Returns:
            Dictionary with message data
        """
        try:
            root = ET.fromstring(xml_content)
            message_data = {}
            for child in root:
                message_data[child.tag] = child.text
            return message_data
        except Exception as e:
            print(f"Error parsing XML message: {e}")
            return {}

    async def handle_webhook_callback(self, query_params: Dict[str, str], body: str) -> str:
        """Handle WeChat webhook callback.
        
        Args:
            query_params: Query parameters (msg_signature, timestamp, nonce, echostr)
            body: Request body (encrypted message)
            
        Returns:
            Response to send to WeChat
        """
        msg_signature = query_params.get("msg_signature", "")
        timestamp = query_params.get("timestamp", "")
        nonce = query_params.get("nonce", "")
        echostr = query_params.get("echostr", "")

        # Verify signature
        if not await self.verify_signature(msg_signature, timestamp, nonce):
            print("Invalid signature")
            return "invalid signature"

        # If echostr is present, this is a validation request
        if echostr:
            try:
                decrypted = self._decrypt_message(echostr)
                return decrypted
            except Exception as e:
                print(f"Failed to decrypt echostr: {e}")
                return "error"

        # Decrypt and parse message
        try:
            encrypted_msg = query_params.get("echostr") or body
            if not encrypted_msg:
                # Try to extract from body
                root = ET.fromstring(body)
                encrypted_msg = root.findtext("Encrypt")
            
            if not encrypted_msg:
                return "error"

            decrypted = self._decrypt_message(encrypted_msg)
            message_data = self._parse_xml_message(decrypted)

            # Create inbound message
            inbound = InboundMessage(
                channel=self.channel_id,
                chat_id=message_data.get("FromUserID", ""),
                user_id=message_data.get("FromUserID", ""),
                content=message_data.get("Content", ""),
                message_id=message_data.get("MsgId", ""),
                metadata={
                    "msg_type": message_data.get("MsgType"),
                    "create_time": message_data.get("CreateTime"),
                    "from_user": message_data.get("FromUserID"),
                    "to_user": message_data.get("ToUserID"),
                    "agent_id": message_data.get("AgentID"),
                },
            )

            await self.handle_inbound(inbound)
            return "success"

        except Exception as e:
            print(f"Error handling webhook callback: {e}")
            return "error"

    async def _get_access_token(self) -> str:
        """Get or refresh WeChat access token."""
        if self._access_token and time.time() < self._token_expires_at - 300:
            return self._access_token

        payload = {
            "corpid": self.corp_id,
            "corpsecret": self.corp_secret,
        }

        result = await self._api_request(
            "gettoken",
            method="GET",
            payload=payload,
            use_token=False,
        )

        self._access_token = result.get("access_token")
        expire = result.get("expires_in", 7200)
        self._token_expires_at = time.time() + expire

        return self._access_token

    async def _api_request(
        self,
        api: str,
        method: str = "GET",
        payload: dict = None,
        use_token: bool = True,
    ) -> dict:
        """Make API request to WeChat.
        
        Args:
            api: API endpoint (without base URL)
            method: HTTP method (GET or POST)
            payload: Request payload
            use_token: Whether to include access token
            
        Returns:
            API response as dict
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        url = f"https://qyapi.weixin.qq.com/cgi-bin/{api}"
        
        if use_token:
            await self._get_access_token()
            if "?" in url:
                url += f"&access_token={self._access_token}"
            else:
                url += f"?access_token={self._access_token}"
        
        try:
            if method == "GET":
                params = payload if payload else {}
                async with self._session.get(url, params=params) as response:
                    result = await response.json()
            else:
                async with self._session.post(url, json=payload) as response:
                    result = await response.json()

            # Check for WeChat API errors
            if result.get("errcode") and result.get("errcode") != 0:
                raise Exception(f"WeChat API error: [{result.get('errcode')}] {result.get('errmsg')}")

            return result
        except Exception as e:
            print(f"Error making WeChat API request: {e}")
            raise

    async def handle_inbound_callback(self, callback_data: dict) -> InboundMessage:
        """Handle incoming message from WeChat webhook callback.
        
        DEPRECATED: Use handle_webhook_callback instead. This method is kept
        for backward compatibility.
        
        Args:
            callback_data: Parsed callback data from WeChat
            
        Returns:
            InboundMessage object
        """
        inbound = InboundMessage(
            channel=self.channel_id,
            chat_id=callback_data.get("FromUserID", ""),
            user_id=callback_data.get("FromUserID", ""),
            content=callback_data.get("Content", ""),
            message_id=callback_data.get("MsgId", ""),
            metadata={
                "msg_type": callback_data.get("MsgType"),
                "create_time": callback_data.get("CreateTime"),
                "from_user": callback_data.get("FromUserID"),
                "to_user": callback_data.get("ToUserID"),
            },
        )
        
        await self.handle_inbound(inbound)
        return inbound


class WeChatWebhookChannel(Channel):
    """WeChat Webhook channel for incoming messages (webhook only)."""

    def __init__(self, config: dict):
        """Initialize WeChat webhook channel.

        Args:
            config: Configuration with 'webhook_url' or webhook callback configuration
        """
        super().__init__(ChannelType.WECHAT, config)
        self.webhook_url = config.get("webhook_url")
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        """Start webhook channel."""
        self._session = aiohttp.ClientSession()
        print("WeChat webhook channel ready")

    async def stop(self) -> None:
        """Stop webhook channel."""
        if self._session:
            await self._session.close()
            self._session = None

    async def send_message(self, message: OutboundMessage) -> str:
        """Send message via custom webhook.
        
        Note: This is typically used for custom routing, not standard WeChat messaging.
        For standard messaging, use WeChatChannel instead.
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        if not self.webhook_url:
            raise ValueError("Webhook URL is required")

        payload = {
            "touser": message.chat_id,
            "msgtype": "text",
            "text": {"content": message.content},
        }

        async with self._session.post(self.webhook_url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Webhook error: {response.status}")
            result = await response.json()

        return result.get("msgid", "webhook_sent")
